#!/usr/bin/env python3
"""Post-cleanup single-op validation for custom fused_infer_attention_score.

Self-contained: fp32 reference for dense/causal/GQA/paged-FD cases,
BlasstGoldenTND (kernel-granularity) for sparse-lambda cases.

Usage: python3 validate_post_cleanup.py [--perf-only] [--prec-only]
"""
import argparse
import math
import os
import sys

_CUR = os.path.dirname(os.path.realpath(__file__))
_CUSTOM_OPP = os.path.join(_CUR, "..", "..", "vllm_ascend", "_cann_ops_custom", "vendors", "custom_transformer")
if os.path.exists(_CUSTOM_OPP):
    os.environ.setdefault("ASCEND_CUSTOM_OPP_PATH", _CUSTOM_OPP)

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import vllm_ascend  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

sys.path.insert(0, _CUR)
from blasst_golden_tnd import BlasstGoldenTND  # noqa: E402

SWA_INT_MAX = 2147483647
DENSE_LAMBDA = -99.0
DEVICE = "npu:0"

# sparse_mode=3 requires a real mask; production passes this int8 compact
# triU mask (vllm_ascend/attention/attention_mask.py get_splitfuse_attn_mask,
# kernel reads it as int8 RowMajor [2048, 2048], 1 = masked position).
_CAUSAL_MASK = None


def causal_mask():
    global _CAUSAL_MASK
    if _CAUSAL_MASK is None:
        _CAUSAL_MASK = torch.triu(
            torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(DEVICE)
    return _CAUSAL_MASK


# ---------------------------------------------------------------------
# Custom op wrapper
# ---------------------------------------------------------------------
def _cumsum(lens):
    out, s = [], 0
    for x in lens:
        s += x
        out.append(s)
    return out


def run_custom(q, k, v, q_lens, kv_lens, num_heads, num_kv_heads, scale,
               sparse_lambda=DENSE_LAMBDA, causal=False, blocktable=None,
               block_size=0, softmax_lse_flag=True, sparse_stats_flag=False):
    # Op contract (matches attention_v1.py): q lens are prefix sums; kv lens
    # are prefix sums for non-paged, raw per-batch lengths for paged.
    paged = blocktable is not None
    return torch.ops._C_ascend.npu_fused_infer_attention_score(
        q, k, v,
        pse_shift=None, atten_mask=causal_mask() if causal else None,
        actual_seq_lengths=_cumsum(q_lens),
        actual_seq_lengths_kv=kv_lens if paged else _cumsum(kv_lens),
        blocktable=blocktable,
        num_heads=num_heads, scale=scale,
        pre_tokens=SWA_INT_MAX, next_tokens=SWA_INT_MAX,
        input_layout="TND", num_key_value_heads=num_kv_heads,
        sparse_mode=3 if causal else 0, inner_precise=0,
        block_size=block_size, antiquant_mode=0,
        sparse_lambda=sparse_lambda, softmax_lse_flag=softmax_lse_flag,
        sparse_stats_flag=sparse_stats_flag,
    )


# ---------------------------------------------------------------------
# fp32 reference (dense semantics; sparse cases use BlasstGoldenTND)
# ---------------------------------------------------------------------
def ref_attention(q, k, v, q_lens, kv_lens, num_heads, num_kv_heads, scale,
                  causal=False, blocktable=None, block_size=0):
    """q: [total_q, H, D]; non-paged k/v: [total_kv, Hkv, D];
    paged k/v: [num_blocks, block_size, Hkv, D] with blocktable [B, max_blocks].
    Returns out [total_q, H, D] in fp32."""
    H, D = q.shape[1], q.shape[2]
    group = num_heads // num_kv_heads if num_kv_heads > 0 else 1
    outs = []
    q_off = 0
    for b, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
        qb = q[q_off:q_off + ql].float()  # [ql, H, D]
        q_off += ql
        if blocktable is not None:
            nb = (kl + block_size - 1) // block_size
            blk = blocktable[b, :nb].long()
            # paged cache is [num_blocks, block_size, Hkv*D]
            kb = k[blk].reshape(-1, num_kv_heads, D)[:kl].float()  # [kl, Hkv, D]
            vb = v[blk].reshape(-1, num_kv_heads, D)[:kl].float()
        else:
            kv_off = int(sum(kv_lens[:b]))
            kb = k[kv_off:kv_off + kl].float()
            vb = v[kv_off:kv_off + kl].float()
        kb = kb.repeat_interleave(group, dim=1)  # [kl, H, D]
        vb = vb.repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qb, kb) * scale  # [H, ql, kl]
        if causal:
            i = torch.arange(ql, device=q.device).unsqueeze(1)
            j = torch.arange(kl, device=q.device).unsqueeze(0)
            mask = j > (i + (kl - ql))
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        ob = torch.einsum("hqk,khd->qhd", probs, vb)
        outs.append(ob)
    return torch.cat(outs, dim=0)


def metrics(out, ref):
    a, b = out.float().flatten(), ref.float().flatten()
    diff = (a - b).abs()
    return {
        "max_err": diff.max().item(),
        "rel_err": (torch.norm(a - b) / (torch.norm(b) + 1e-9)).item(),
        "cos_sim": F.cosine_similarity(a, b, dim=0).item(),
    }


def tol(dtype):
    return (0.02, 0.995) if dtype == torch.float16 else (0.03, 0.99)


# ---------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------
def make_varlen(q_lens, kv_lens, H, Hkv, D, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    tq, tkv = sum(q_lens), sum(kv_lens)
    q = torch.randn(tq, H, D, generator=g, dtype=torch.float32).to(dtype).to(DEVICE)
    k = torch.randn(tkv, Hkv, D, generator=g, dtype=torch.float32).to(dtype).to(DEVICE)
    v = torch.randn(tkv, Hkv, D, generator=g, dtype=torch.float32).to(dtype).to(DEVICE)
    return q, k, v


def make_paged(kv_lens, H, D, block_size, dtype, seed=0):
    # Production paged layout: [num_blocks, block_size, Hkv*D] (3D view).
    g = torch.Generator(device="cpu").manual_seed(seed)
    max_blocks = max((kl + block_size - 1) // block_size for kl in kv_lens)
    num_blocks = sum((kl + block_size - 1) // block_size for kl in kv_lens)
    k = torch.randn(num_blocks, block_size, H * D, generator=g, dtype=torch.float32).to(dtype).to(DEVICE)
    v = torch.randn(num_blocks, block_size, H * D, generator=g, dtype=torch.float32).to(dtype).to(DEVICE)
    bt = torch.zeros(len(kv_lens), max_blocks, dtype=torch.int32)
    off = 0
    for b, kl in enumerate(kv_lens):
        nb = (kl + block_size - 1) // block_size
        bt[b, :nb] = torch.arange(off, off + nb, dtype=torch.int32)
        off += nb
    return k, v, bt.to(DEVICE)


# ---------------------------------------------------------------------
# Precision cases
# ---------------------------------------------------------------------
def run_precision():
    results = []
    H, D = 16, 128
    scale = 1.0 / math.sqrt(D)

    def check(name, out, ref, dtype):
        m = metrics(out, ref)
        rel_lim, cos_lim = tol(dtype)
        ok = m["rel_err"] <= rel_lim and m["cos_sim"] >= cos_lim
        results.append((name, ok, m))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: rel_err={m['rel_err']:.4e} "
              f"max_err={m['max_err']:.4e} cos={m['cos_sim']:.6f}")

    for dtype in (torch.float16, torch.bfloat16):
        tag = "fp16" if dtype == torch.float16 else "bf16"

        # 1. dense prefill, no mask
        q_lens = kv_lens = [257, 1024, 33]
        q, k, v = make_varlen(q_lens, kv_lens, H, H, D, dtype, seed=1)
        out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, H, scale)
        ref = ref_attention(q, k, v, q_lens, kv_lens, H, H, scale)
        check(f"{tag}/dense-nomask", out, ref, dtype)

        # 2. dense prefill, causal
        q_lens = kv_lens = [300, 777]
        q, k, v = make_varlen(q_lens, kv_lens, H, H, D, dtype, seed=2)
        out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, H, scale, causal=True)
        ref = ref_attention(q, k, v, q_lens, kv_lens, H, H, scale, causal=True)
        check(f"{tag}/dense-causal", out, ref, dtype)

        # 3. GQA (Hkv=4)
        Hkv = 4
        q_lens = kv_lens = [128, 512]
        q, k, v = make_varlen(q_lens, kv_lens, H, Hkv, D, dtype, seed=3)
        out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, Hkv, scale)
        ref = ref_attention(q, k, v, q_lens, kv_lens, H, Hkv, scale)
        check(f"{tag}/gqa", out, ref, dtype)

        # 4. paged decode q=1, kv=4096 -> FD path (numTasks<=8, kv>=4096).
        # FD requires lseFlag=False and blockSize=128 (tiling.cpp FD gate).
        bs, Hkv, blk = 2, 4, 128
        q_lens = [1] * bs
        kv_lens = [4096, 5000]
        q, _, _ = make_varlen(q_lens, [0] * bs, H, Hkv, D, dtype, seed=4)
        k, v, bt = make_paged(kv_lens, Hkv, D, blk, dtype, seed=4)
        out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, Hkv, scale,
                               blocktable=bt, block_size=blk, softmax_lse_flag=False)
        ref = ref_attention(q, k, v, q_lens, kv_lens, H, Hkv, scale,
                            blocktable=bt, block_size=blk)
        check(f"{tag}/paged-decode-fd", out, ref, dtype)

        # 5. paged decode, short kv -> regular paged path (no FD)
        kv_lens = [100, 300]
        q, _, _ = make_varlen(q_lens, [0] * bs, H, Hkv, D, dtype, seed=5)
        k, v, bt = make_paged(kv_lens, Hkv, D, blk, dtype, seed=5)
        out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, Hkv, scale,
                               blocktable=bt, block_size=blk)
        ref = ref_attention(q, k, v, q_lens, kv_lens, H, Hkv, scale,
                            blocktable=bt, block_size=blk)
        check(f"{tag}/paged-decode-regular", out, ref, dtype)

        # 6. sparse blasst, lambda=-3, kernel-granularity golden.
        # q_lens multiples of 128 so every qSBlock tile has clean 16-row
        # rowloops (golden limitation). Batch 2's first KV stack is boosted
        # x4 (large gm) and the remaining stacks scaled to ~0, so skip
        # decisions are unambiguous: stacks 1+ of batch 2 must be skipped.
        lam = -3.0
        q_lens = kv_lens = [640, 1408]
        q, k, v = make_varlen(q_lens, kv_lens, H, H, D, dtype, seed=6)
        b2 = q_lens[0]  # batch 2 kv starts at global row 640
        k[b2:b2 + 512] = (k[b2:b2 + 512].float() * 4.0).to(dtype)
        k[b2 + 512:] = (k[b2 + 512:].float() * 0.001).to(dtype)
        golden = BlasstGoldenTND(num_heads=H, num_key_value_heads=H, head_dim=D,
                                 scale=scale, block_size=32, sparse_lamda=lam,
                                 rowloop_rows=16)
        ref, _, info = golden.forward_blasst_kernel(
            q.float().cpu(), k.float().cpu(), v.float().cpu(),
            torch.tensor(_cumsum(q_lens)), torch.tensor(_cumsum(kv_lens)))

        # 6a. stats mode = detection only (no real skip): compare skip counts
        _, _, stats = run_custom(q, k, v, q_lens, kv_lens, H, H, scale,
                                 sparse_lambda=lam, sparse_stats_flag=True)
        s_skip, s_tot = int(stats[0]), int(stats[1])
        ok = (s_skip == info["skipped_blocks"] and s_tot == info["total_blocks"])
        results.append((f"{tag}/blasst-stats", ok,
                        {"rel_err": 0.0 if ok else 1.0, "max_err": 0.0, "cos_sim": 1.0}))
        print(f"[{'PASS' if ok else 'FAIL'}] {tag}/blasst-stats: "
              f"custom {s_skip}/{s_tot} skipped vs golden "
              f"{info['skipped_blocks']}/{info['total_blocks']}")

        # 6b. real-skip mode: output vs golden
        out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, H, scale, sparse_lambda=lam)
        check(f"{tag}/blasst-lambda-3", out, ref.to(DEVICE), dtype)

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n=== precision: {len(results) - n_fail}/{len(results)} passed ===")
    return n_fail == 0


# ---------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------
def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # us/iter


def run_perf():
    H, D = 16, 128
    scale = 1.0 / math.sqrt(D)
    print("\n=== performance (us/call) ===")

    # prefill: 4 x 2048 tokens, causal
    q_lens = kv_lens = [2048] * 4
    for dtype, tag in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        q, k, v = make_varlen(q_lens, kv_lens, H, H, D, dtype, seed=7)
        t = bench(lambda: run_custom(q, k, v, q_lens, kv_lens, H, H, scale, causal=True))
        print(f"prefill-4x2048-causal/{tag}: custom {t:9.1f} us")

    # decode paged FD: bs=8, kv=8192, q=1, GQA (lseFlag=False to enable FD)
    Hkv, blk = 4, 128
    bs = 8
    kv_lens = [8192] * bs
    q_lens = [1] * bs
    for dtype, tag in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        q, _, _ = make_varlen(q_lens, [0] * bs, H, Hkv, D, dtype, seed=8)
        k, v, bt = make_paged(kv_lens, Hkv, D, blk, dtype, seed=8)
        t = bench(lambda: run_custom(q, k, v, q_lens, kv_lens, H, Hkv, scale,
                                     blocktable=bt, block_size=blk,
                                     softmax_lse_flag=False))
        print(f"decode-fd-bs8-kv8192/{tag}: custom {t:9.1f} us")

    # decode paged regular: bs=8, kv=512
    kv_lens = [512] * bs
    for dtype, tag in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        q, _, _ = make_varlen(q_lens, [0] * bs, H, Hkv, D, dtype, seed=9)
        k, v, bt = make_paged(kv_lens, Hkv, D, blk, dtype, seed=9)
        t = bench(lambda: run_custom(q, k, v, q_lens, kv_lens, H, Hkv, scale,
                                     blocktable=bt, block_size=blk))
        print(f"decode-regular-bs8-kv512/{tag}: custom {t:9.1f} us")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf-only", action="store_true")
    ap.add_argument("--prec-only", action="store_true")
    args = ap.parse_args()
    ok = True
    if not args.perf_only:
        ok = run_precision()
    if not args.prec_only:
        run_perf()
    sys.exit(0 if ok else 1)
