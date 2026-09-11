#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe the NPU kernel's block-count semantics (gLSE counters) on synthetic
TND causal inputs, to pin down the exact skip/counting granularity.

For each config we print the per-core [sparse,total] counters parsed from the
gLSE head, plus the total.  sparse_lambda=-3.0 activates the BlasST path.
"""

import math
import os
import sys

_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
_CUSTOM_OPP_PATH = os.path.join(
    _CUR_DIR, "..", "..", "vllm_ascend", "_cann_ops_custom", "vendors", "vllm-ascend"
)
if os.path.exists(_CUSTOM_OPP_PATH):
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = _CUSTOM_OPP_PATH

import torch


def parse_blocknum_from_lse(lse: torch.Tensor, max_cores: int = 48):
    data = lse.detach().cpu().flatten()
    core_num = min(max_cores, len(data) // 16)
    sp_total = 0
    cnt_total = 0
    per_core = []
    for core in range(core_num):
        sp = int(data[core * 16].item())
        total = int(data[core * 16 + 1].item())
        per_core.append((sp, total))
        sp_total += sp
        cnt_total += total
    return sp_total, cnt_total, per_core


def run_case(name, q_len, kv_len, num_heads, num_kv_heads, head_dim=128,
             causal=True, sparse_lambda=-3.0, device="npu:14"):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    query = torch.randn(q_len, num_heads, head_dim, dtype=dtype, device=device)
    key = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    q_cum = [q_len]
    kv_cum = [kv_len]
    scale = 1.0 / math.sqrt(head_dim)

    atten_mask = None
    if causal:
        # Right-aligned causal mask matching the dump format: 1=masked.
        # Window covers the last q_len kv positions; row i sees window cols <= i.
        win = torch.ones(q_len, q_len, dtype=torch.int8)
        win = torch.triu(win, diagonal=1)
        atten_mask = win.to(device)

    out, lse, _ = torch.ops._C_ascend.npu_fused_infer_attention_score(
        query, key, value,
        pse_shift=None,
        atten_mask=atten_mask,
        actual_seq_lengths=q_cum,
        actual_seq_lengths_kv=kv_cum,
        blocktable=None,
        num_heads=num_heads,
        scale=scale,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        input_layout="TND",
        num_key_value_heads=num_kv_heads,
        sparse_mode=3 if causal else 0,
        inner_precise=0,
        block_size=0,
        antiquant_mode=0,
        sparse_lambda=sparse_lambda,
        softmax_lse_flag=True,
    )
    torch.npu.synchronize()

    sp, cnt, per_core = parse_blocknum_from_lse(lse)
    nonzero = [(c, s, t) for c, (s, t) in enumerate(per_core) if t > 0]
    print(f"[{name}] q={q_len} kv={kv_len} H={num_heads}/{num_kv_heads} "
          f"causal={causal}: sparse={sp} total={cnt} "
          f"rate={(sp / cnt) if cnt else 0:.2%}")
    print(f"    per-core (core,sp,total): {nonzero}")

    # Model A prediction: noSkip=(qb+1)*128+diffS, stacks=ceil(noSkip/512)
    diff_s = max(0, kv_len - q_len) if causal else 0
    model_a = 0
    group = num_heads // num_kv_heads
    qn_tile = max(1, (128 // q_len) // 2 * 2)
    qn_tile = min(qn_tile, group)
    qn_blocks_per_group = (group + qn_tile - 1) // qn_tile
    units = ((q_len + 127) // 128) * qn_blocks_per_group * num_kv_heads
    for qb in range((q_len + 127) // 128):
        no_skip = min(kv_len, (qb + 1) * 128 + diff_s) if causal else kv_len
        stacks = (no_skip + 511) // 512
        model_a += stacks * qn_blocks_per_group * num_kv_heads
    print(f"    model_A total={model_a} (units={units}, qn_tile={qn_tile})")


def main():
    import torch_npu  # noqa: F401
    from vllm_ascend import platform
    platform.NPUPlatform.import_kernels()
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    device = "npu:14"
    torch.npu.set_device(device)

    # Baseline probes mirroring the dump shapes.
    run_case("prefill-like", 2048, 2048, 8, 1, device=device)
    run_case("chunked-like", 2048, 4096, 8, 1, device=device)
    # Smaller probes to triangulate.
    run_case("tiny-512", 512, 512, 2, 1, device=device)
    run_case("small-1024", 1024, 1024, 8, 1, device=device)
    run_case("single-head", 2048, 2048, 1, 1, device=device)
    run_case("nomask", 2048, 2048, 8, 1, causal=False, device=device)
    run_case("half-masked", 2048, 2048, 8, 1, sparse_lambda=-99.0, device=device)


if __name__ == "__main__":
    main()
