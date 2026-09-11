#!/usr/bin/env python3
"""Probe: does the custom FIA op support head_dim=256 (Qwen3.5 family)?

Runs dense-causal / GQA / paged-decode cases at D=256 against the same
reference used by validate_post_cleanup. Decides whether a Qwen3.5 accuracy
run can route full-attention layers through the custom op.
"""
import math
import sys

sys.path.insert(0, "/home/z00603376/fia/vllm-ascend/tests/fused_infer_attention_score")
import torch  # noqa: E402
from validate_post_cleanup import (  # noqa: E402
    DEVICE, run_custom, ref_attention, metrics, make_varlen, make_paged,
)

DT = torch.float16
D = 256
SCALE = 1.0 / math.sqrt(D)


def check(name, out, ref):
    m = metrics(out.float(), ref.float())
    ok = m["rel_err"] <= 0.02 and m["cos_sim"] >= 0.995
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: rel_err={m['rel_err']:.4e} "
          f"cos={m['cos_sim']:.6f} max_err={m['max_err']:.4e}", flush=True)
    return ok


def main():
    ok = True
    q_lens = kv_lens = [257, 1024, 33]

    # 1) dense causal, MHA
    q, k, v = make_varlen(q_lens, kv_lens, 8, 8, D, DT, seed=11)
    out, _, _ = run_custom(q, k, v, q_lens, kv_lens, 8, 8, SCALE, causal=True)
    ref = ref_attention(q, k, v, q_lens, kv_lens, 8, 8, SCALE, causal=True)
    ok &= check("D256/dense-causal-mha", out, ref)

    # 2) GQA 12:2 (Qwen3.5 ratio 6:1)
    q, k, v = make_varlen(q_lens, kv_lens, 12, 2, D, DT, seed=12)
    out, _, _ = run_custom(q, k, v, q_lens, kv_lens, 12, 2, SCALE, causal=True)
    ref = ref_attention(q, k, v, q_lens, kv_lens, 12, 2, SCALE, causal=True)
    ok &= check("D256/gqa-12-2", out, ref)

    # 3) paged decode (FD path: kv>=4096, blockSize=128, lse off)
    bs, Hkv, blk = 2, 2, 128
    q, _, _ = make_varlen([1] * bs, [0] * bs, 6, Hkv, D, DT, seed=13)
    k, v, bt = make_paged([4096, 5000], Hkv, D, blk, DT, seed=13)
    out, _, _ = run_custom(q, k, v, [1] * bs, [4096, 5000], 6, Hkv, SCALE,
                           blocktable=bt, block_size=blk, softmax_lse_flag=False)
    ref = ref_attention(q, k, v, [1] * bs, [4096, 5000], 6, Hkv, SCALE,
                        blocktable=bt, block_size=blk)
    ok &= check("D256/paged-decode-fd", out, ref)

    print("VERDICT:", "head_dim=256 SUPPORTED" if ok else "head_dim=256 NOT SUPPORTED")


if __name__ == "__main__":
    main()
