#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Localize residual diffs between NPU (lambda=-3) and the kernel-granularity
golden on the ChunkedPrefill dump: which (qSBlock, head, rowLoop) units carry
the largest errors, and what the local skip patterns look like.
"""

import os
import sys

_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
_CUSTOM_OPP_PATH = os.path.join(
    _CUR_DIR, "..", "..", "vllm_ascend", "_cann_ops_custom", "vendors", "vllm-ascend"
)
if os.path.exists(_CUSTOM_OPP_PATH):
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = _CUSTOM_OPP_PATH

import torch

sys.path.insert(0, _CUR_DIR)
from test_fia_dump_cases import (  # noqa: E402
    load_dump, build_call_kwargs, run_golden_blasst, reconstruct_golden_inputs)


def main():
    import torch_npu  # noqa: F401
    from vllm_ascend import platform
    platform.NPUPlatform.import_kernels()
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    device = torch.device("npu:14")
    torch.npu.set_device(device)

    dump = ("/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps/"
            "fia_layer30_rank0_stateChunkedPrefill.pt")
    meta, inputs, dumped_output = load_dump(dump)

    # NPU run at lambda=-3
    kwargs, sparse_lambda = build_call_kwargs(inputs, device, -3.0)
    with torch.npu.device(device):
        out_npu, lse_npu = torch.ops._C_ascend.npu_fused_infer_attention_score(
            kwargs["query"], kwargs["key"], kwargs["value"],
            pse_shift=None, atten_mask=kwargs.get("atten_mask"),
            actual_seq_lengths=kwargs["actual_seq_lengths"],
            actual_seq_lengths_kv=kwargs["actual_seq_lengths_kv"],
            blocktable=kwargs.get("block_table"),
            num_heads=kwargs["num_heads"], scale=kwargs["scale"],
            pre_tokens=kwargs["pre_tokens"], next_tokens=kwargs["next_tokens"],
            input_layout=kwargs["input_layout"],
            num_key_value_heads=kwargs["num_key_value_heads"],
            sparse_mode=kwargs["sparse_mode"],
            inner_precise=kwargs.get("inner_precise", 0),
            block_size=kwargs["block_size"], antiquant_mode=0,
            sparse_lambda=sparse_lambda, softmax_lse_flag=True)
        torch.npu.synchronize()

    out_golden, info = run_golden_blasst(inputs, sparse_lambda,
                                         golden_mode="kernel")
    print(f"golden sparsity: {info['skipped_blocks']}/{info['total_blocks']} "
          f"= {info['sparsity']:.2%}")

    diff = (out_npu.cpu().float() - out_golden.float()).abs()  # (T, H, D)
    row_diff = diff.max(dim=-1).values  # (T, H)
    print(f"overall: max={diff.max().item():.6e} mean={diff.mean().item():.6e}")
    for thr in (0.05, 0.02, 0.01, 0.005):
        n = int((row_diff > thr).sum().item())
        print(f"  rows with |diff|>{thr}: {n} / {row_diff.numel()}")

    top = torch.topk(row_diff.flatten(), 12)
    print("\ntop-12 (row, head, qSBlock, rowLoop, diff):")
    for val, idx in zip(top.values, top.indices):
        t = int(idx) // row_diff.size(1)
        h = int(idx) % row_diff.size(1)
        print(f"  row={t:5d} head={h} qb={t // 128:2d} rowLoop={(t % 128) // 16} "
              f"diff={val.item():.6e}")

    # Per (qb, head) max diff heat summary
    print("\nper-(qb, head) max diff (rows with >0.02 marked *):")
    qb_h = row_diff.view(16, 128, 8).max(dim=1).values  # (16, 8)
    for qb in range(16):
        line = " ".join(f"{qb_h[qb, h].item():.3f}" for qb in [qb] for h in range(8))
        flags = " ".join("*" if qb_h[qb, h].item() > 0.02 else "." for h in range(8))
        print(f"  qb={qb:2d}: {line}   {flags}")


if __name__ == "__main__":
    main()
