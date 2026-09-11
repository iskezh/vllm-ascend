#!/usr/bin/env python3
"""Faithful bisect driver for the in-suite 6b (blasst-lambda-3) crash.

Exact copy of validate_post_cleanup.run_precision()'s fp16 half, including
every metrics()/check() device op, but with cases selectable so the poisoning
predecessor can be bisected.

Usage: python3 repro_6b.py 1 2 3 4 5 6a 6b
"""
import math
import sys

sys.path.insert(0, "/home/z00603376/fia/vllm-ascend/tests/fused_infer_attention_score")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from validate_post_cleanup import (  # noqa: E402
    DEVICE, run_custom, ref_attention, metrics, tol,
    make_varlen, make_paged, _cumsum,
)
from blasst_golden_tnd import BlasstGoldenTND  # noqa: E402

H, D = 16, 128
SCALE = 1.0 / math.sqrt(D)
DT = torch.float16
TAG = "fp16"


def check(name, out, ref):
    m = metrics(out, ref)
    ok = m["rel_err"] <= 0.02 and m["cos_sim"] >= 0.995
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: rel_err={m['rel_err']:.4e} cos={m['cos_sim']:.6f}",
          flush=True)


def case1():
    q_lens = kv_lens = [257, 1024, 33]
    q, k, v = make_varlen(q_lens, kv_lens, H, H, D, DT, seed=1)
    out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, H, SCALE)
    ref = ref_attention(q, k, v, q_lens, kv_lens, H, H, SCALE)
    check(f"{TAG}/dense-nomask", out, ref)


def case2():
    q_lens = kv_lens = [300, 777]
    q, k, v = make_varlen(q_lens, kv_lens, H, H, D, DT, seed=2)
    out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, H, SCALE, causal=True)
    ref = ref_attention(q, k, v, q_lens, kv_lens, H, H, SCALE, causal=True)
    check(f"{TAG}/dense-causal", out, ref)


def case3():
    Hkv = 4
    q_lens = kv_lens = [128, 512]
    q, k, v = make_varlen(q_lens, kv_lens, H, Hkv, D, DT, seed=3)
    out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, Hkv, SCALE)
    ref = ref_attention(q, k, v, q_lens, kv_lens, H, Hkv, SCALE)
    check(f"{TAG}/gqa", out, ref)


def case4():
    bs, Hkv, blk = 2, 4, 128
    q_lens = [1] * bs
    kv_lens = [4096, 5000]
    q, _, _ = make_varlen(q_lens, [0] * bs, H, Hkv, D, DT, seed=4)
    k, v, bt = make_paged(kv_lens, Hkv, D, blk, DT, seed=4)
    out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, Hkv, SCALE,
                           blocktable=bt, block_size=blk, softmax_lse_flag=False)
    ref = ref_attention(q, k, v, q_lens, kv_lens, H, Hkv, SCALE,
                        blocktable=bt, block_size=blk)
    check(f"{TAG}/paged-decode-fd", out, ref)


def case5():
    bs, Hkv, blk = 2, 4, 128
    kv_lens = [100, 300]
    q_lens = [1] * bs
    q, _, _ = make_varlen(q_lens, [0] * bs, H, Hkv, D, DT, seed=5)
    k, v, bt = make_paged(kv_lens, Hkv, D, blk, DT, seed=5)
    out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, Hkv, SCALE,
                           blocktable=bt, block_size=blk)
    ref = ref_attention(q, k, v, q_lens, kv_lens, H, Hkv, SCALE,
                        blocktable=bt, block_size=blk)
    check(f"{TAG}/paged-decode-regular", out, ref)


def case6_input():
    lam = -3.0
    q_lens = kv_lens = [640, 1408]
    q, k, v = make_varlen(q_lens, kv_lens, H, H, D, DT, seed=6)
    b2 = q_lens[0]
    k[b2:b2 + 512] = (k[b2:b2 + 512].float() * 4.0).to(DT)
    k[b2 + 512:] = (k[b2 + 512:].float() * 0.001).to(DT)
    return q, k, v, q_lens, kv_lens, lam


def case6a():
    q, k, v, q_lens, kv_lens, lam = case6_input()
    golden = BlasstGoldenTND(num_heads=H, num_key_value_heads=H, head_dim=D,
                             scale=SCALE, block_size=32, sparse_lamda=lam,
                             rowloop_rows=16)
    ref, _, info = golden.forward_blasst_kernel(
        q.float().cpu(), k.float().cpu(), v.float().cpu(),
        torch.tensor(_cumsum(q_lens)), torch.tensor(_cumsum(kv_lens)))
    _, _, stats = run_custom(q, k, v, q_lens, kv_lens, H, H, SCALE,
                             sparse_lambda=lam, sparse_stats_flag=True)
    s_skip, s_tot = int(stats[0]), int(stats[1])
    ok = (s_skip == info["skipped_blocks"] and s_tot == info["total_blocks"])
    print(f"[{'PASS' if ok else 'FAIL'}] {TAG}/blasst-stats: "
          f"custom {s_skip}/{s_tot} vs golden {info['skipped_blocks']}/{info['total_blocks']}",
          flush=True)
    globals()["_ref6"] = ref


def case6b():
    q, k, v, q_lens, kv_lens, lam = case6_input()
    out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, H, SCALE,
                           sparse_lambda=lam)
    check(f"{TAG}/blasst-lambda-3", out, globals()["_ref6"].to(DEVICE))
    print("    6b survived", flush=True)


CASES = {"1": case1, "2": case2, "3": case3, "4": case4, "5": case5,
         "6a": case6a, "6b": case6b}


def main():
    ids = sys.argv[1:]
    print(f"running: {ids}", flush=True)
    for cid in ids:
        print(f"  case {cid} ...", flush=True)
        CASES[cid]()
    print("ALL DONE, no crash", flush=True)


if __name__ == "__main__":
    main()
