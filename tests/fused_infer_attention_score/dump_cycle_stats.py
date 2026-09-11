#!/usr/bin/env python3
"""Per-core cycle counter dump for custom fused_infer_attention_score.

Runs the op with sparse_stats_flag=True and decodes the debug slots written
into the sparse_stats output tensor (int32, 4096 elems). Layout (see
op_kernel/fused_infer_attention_score_kernel.h ~L446):
  stats[0]=sparse_block_sum, stats[1]=total_block_sum   (cube core0, aggregate)
  cube core c: stats[64 + c*32 + 0..5]
    = taskCycles, loadQ, qk, pv, stacks, tasks
  vec core c sub b: stats[2112 + c*64 + b*32 + 0..7]
    = taskCycles, waitQk, softmax, spFlag, waitPv, rescale, stacks, tasks
Counters are per-launch cumulative totals, clamped to int32. Non-FD path only.

Usage: python3 dump_cycle_stats.py [--shape sparse|prefill]
"""
import argparse
import math
import os
import sys

_CUR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _CUR)

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

from validate_post_cleanup import (  # noqa: E402
    DEVICE, run_custom, make_varlen,
)

CUBE_BASE, VEC_BASE = 64, 2112
CUBE_NAMES = ["taskCycles", "loadQ", "qk", "pv", "stacks", "tasks"]
VEC_NAMES = ["taskCycles", "waitQk", "softmax", "spFlag", "waitPv",
             "rescale", "stacks", "tasks"]
MAX_CORES = 30  # slot layout valid for coreNum <= 30


def dump(stats, title):
    s = stats.cpu().tolist()
    print(f"\n=== {title} ===")
    print(f"aggregate: sparse_blocks={s[0]} total_blocks={s[1]}")

    print("-- cube cores --")
    rows = []
    for c in range(MAX_CORES):
        vals = s[CUBE_BASE + c * 32: CUBE_BASE + c * 32 + 6]
        if any(vals):
            rows.append((c, vals))
    if not rows:
        print("(all zero)")
    for c, vals in rows:
        print(f"cube[{c:2d}] " + " ".join(
            f"{n}={v}" for n, v in zip(CUBE_NAMES, vals)))
    tot = [sum(r[1][i] for r in rows) for i in range(6)] if rows else [0] * 6
    print("cube  Σ  " + " ".join(f"{n}={v}" for n, v in zip(CUBE_NAMES, tot)))

    print("-- vec cores (sub-block level) --")
    rows = []
    for c in range(MAX_CORES):
        for b in range(2):
            off = VEC_BASE + c * 64 + b * 32
            vals = s[off:off + 8]
            if any(vals):
                rows.append((c, b, vals))
    if not rows:
        print("(all zero)")
    for c, b, vals in rows:
        print(f"vec[{c:2d}.{b}] " + " ".join(
            f"{n}={v}" for n, v in zip(VEC_NAMES, vals)))
    tot = [sum(r[2][i] for r in rows) for i in range(8)] if rows else [0] * 8
    print("vec   Σ  " + " ".join(f"{n}={v}" for n, v in zip(VEC_NAMES, tot)))


def run_sparse():
    """Case-6 shape from validate_post_cleanup: skip decisions unambiguous
    (batch2 first KV stack boosted x4, rest scaled to ~0)."""
    H, D = 16, 128
    scale = 1.0 / math.sqrt(D)
    lam = -3.0
    q_lens = kv_lens = [640, 1408]
    q, k, v = make_varlen(q_lens, kv_lens, H, H, D, torch.float16, seed=6)
    b2 = q_lens[0]
    k[b2:b2 + 512] = (k[b2:b2 + 512].float() * 4.0).to(torch.float16)
    k[b2 + 512:] = (k[b2 + 512:].float() * 0.001).to(torch.float16)
    _, _, stats = run_custom(q, k, v, q_lens, kv_lens, H, H, scale,
                             sparse_lambda=lam, sparse_stats_flag=True)
    return stats, f"sparse lam=-3 q/kv={q_lens} (fp16)"


def run_prefill():
    """Perf-shape causal prefill (dense path, stats mode is detection-only)."""
    H, D = 16, 128
    scale = 1.0 / math.sqrt(D)
    q_lens = kv_lens = [2048] * 4
    q, k, v = make_varlen(q_lens, kv_lens, H, H, D, torch.float16, seed=7)
    _, _, stats = run_custom(q, k, v, q_lens, kv_lens, H, H, scale,
                             causal=True, sparse_stats_flag=True)
    return stats, "causal prefill 4x2048 (fp16, dense)"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=["sparse", "prefill", "both"],
                    default="both")
    args = ap.parse_args()
    shapes = [run_sparse, run_prefill]
    if args.shape != "both":
        shapes = [run_sparse] if args.shape == "sparse" else [run_prefill]
    for fn in shapes:
        stats, title = fn()
        dump(stats, title)
