#!/usr/bin/env python3
"""custom（vllm-ascend） vs base（ops-transformer-dev / torch_npu FIA）单算子矩阵对比。

维度：
  - dump 数据：prefill_l0 / prefill_l30 / decode_l20（真实 dump）
  - 稀疏阈值：-99 / -7 / -5 / -3 / -1
  - 场景：稀疏打印开（stats=True，只检测不跳）/ 关（stats=False，真跳过）
  - 对比项：output 精度（max_abs / mean_abs）+ 稀疏结果（custom sparse_stats
    vs base LSE 解析的 blockSparseCount/blockCount）

用法：ASCEND_RT_VISIBLE_DEVICES=4 python test_custom_vs_base_matrix.py
"""

import os
import sys

import torch

_CUR = os.path.dirname(os.path.realpath(__file__))
_CUSTOM_OPP = os.path.join(_CUR, "..", "..", "vllm_ascend", "_cann_ops_custom", "vendors", "vllm-ascend")
if os.path.exists(_CUSTOM_OPP):
    os.environ.setdefault("ASCEND_CUSTOM_OPP_PATH", _CUSTOM_OPP)

import torch_npu  # noqa: F401
import vllm_ascend  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

sys.path.insert(0, _CUR)
from run_dump_repro import _to_tensor, load_dump, run_custom_op, run_torchnpu_op  # noqa: E402
from test_fia_dump_cases import parse_blocknum_from_lse  # noqa: E402

CASES = [
    ("prefill_l0", "/home/z00603376/blasst_model/kvcomp/glse_dump/chunked_prefill_dense_1785123370153_pid1615870.pt"),
    ("prefill_l30", "/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps_v2/fia_layer30_rank0_chunked_prefill_occ1.pt"),
    ("decode_l20", "/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps_v2/fia_layer20_rank0_decode_bigdiff_occ0.pt"),
]
LAMBDAS = [-99.0, -7.0, -5.0, -3.0, -1.0]


def main():
    dev = "npu:0"
    print("=" * 118)
    print(f"custom vs base 矩阵对比  device={dev}  visible={os.environ.get('ASCEND_RT_VISIBLE_DEVICES','?')}")
    print("=" * 118)

    for case_name, path in CASES:
        inputs, params, _, _, meta = load_dump(path)
        NH, NKV = params["num_heads"], params["num_kv_heads"]
        SC, BS = params["scale"], params["block_size"]
        sm = int(params.get("sparse_mode", 3))

        q = inputs["query"].to(dev)
        k = inputs["key"].to(dev)
        v = inputs["value"].to(dev)
        am = inputs["atten_mask"]
        am = am.to(dev) if isinstance(am, torch.Tensor) and am.numel() > 0 else None
        aq = _to_tensor(inputs["actual_seq_lengths_q"], dev)
        akv = _to_tensor(inputs["actual_seq_lengths_kv"], dev)
        bt = inputs["block_table"]
        bt = bt.to(dev) if isinstance(bt, torch.Tensor) else None
        if sm == 3 and am is None:
            sm = 0

        # warmup
        for _ in range(3):
            run_custom_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, -99.0,
                          lse_flag=False, stats_flag=False)
        torch.npu.synchronize()

        print(f"\ncase: {case_name}  q={list(q.shape)}  NH={NH} NKV={NKV} sm={sm}")
        print(f"  {'lambda':>6} | {'场景':>8} | {'out max':>10} {'out mean':>10} | "
              f"{'custom sparse/total':>20} | {'base sparse/total':>18}")

        for lam in LAMBDAS:
            # ---- 稀疏打印关：custom 真跳过 vs base(antiquant 稀疏) ----
            out_c, _, _ = run_custom_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, lam,
                                        lse_flag=False, stats_flag=False)
            torch.npu.synchronize()
            out_b, lse_b = run_torchnpu_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, lam, True)
            torch.npu.synchronize()
            d = (out_c.cpu().float() - out_b.cpu().float()).abs()
            bsp, btot, _ = parse_blocknum_from_lse(lse_b)
            print(f"  {lam:>6.1f} | {'关':>8} | {d.max().item():>10.4e} {d.mean().item():>10.4e} | "
                  f"{'-/-':>20} | {f'{bsp}/{btot}':>18}")

            # ---- 稀疏打印开：custom 检测式（输出=dense） + sparse_stats ----
            out_c2, _, ss = run_custom_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, lam,
                                          lse_flag=False, stats_flag=True)
            torch.npu.synchronize()
            d2 = (out_c2.cpu().float() - out_b.cpu().float()).abs()
            vv = ss.cpu().flatten()
            sp, tot = int(vv[0]), int(vv[1])
            print(f"  {lam:>6.1f} | {'开':>8} | {d2.max().item():>10.4e} {d2.mean().item():>10.4e} | "
                  f"{f'{sp}/{tot}':>20} | {f'{bsp}/{btot}':>18}")

    print("\n" + "=" * 118)
    print("完成")
    print("=" * 118)


if __name__ == "__main__":
    main()
