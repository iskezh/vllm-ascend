#!/usr/bin/env python3
"""单算子性能对比：真实 dump 数据（prefill + decode）× torch_npu × custom 多阈值。

对比矩阵：
  - baseline: torch_npu 线上路径（dense, lse=False）
  - custom: lambda ∈ [-99, -7, -5, -3, -1] × {不统计(stats=False), 统计(stats=True)}
统计场景额外输出 sparse_stats（sparseblock/totalblock → 检测稀疏率）。

用法: python test_perf_dump_cases.py [--iters N] [--warmup M] [--device npu:0]
"""

import argparse
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
from run_dump_repro import _bench_one, _to_tensor, load_dump, run_custom_op, run_torchnpu_faithful  # noqa: E402

CASES = [
    ("decode_l20", "/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps_v2/fia_layer20_rank0_decode_bigdiff_occ0.pt"),
    ("prefill_l30", "/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps_v2/fia_layer30_rank0_chunked_prefill_occ1.pt"),
    ("prefill_l0", "/home/z00603376/blasst_model/kvcomp/glse_dump/chunked_prefill_dense_1785123370153_pid1615870.pt"),
]
LAMBDAS = [-99.0, -7.0, -5.0, -3.0, -1.0]


def parse_sparse_stats(ss):
    try:
        v = ss.cpu().flatten()
        return int(v[0].item()), int(v[1].item())
    except Exception:
        return -1, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default="npu:0")
    args = ap.parse_args()

    print("=" * 128)
    print(f"单算子性能对比（真实 dump 数据）  iters={args.iters} warmup={args.warmup} device={args.device}")
    print("=" * 128)

    for case_name, path in CASES:
        inputs, params, _, _, meta = load_dump(path)
        NH, NKV = params["num_heads"], params["num_kv_heads"]
        SC, BS = params["scale"], params["block_size"]
        sm = int(params.get("sparse_mode", 3))
        lam0 = float(params.get("sparse_lambda", -99.0))

        q = inputs["query"].to(args.device)
        k = inputs["key"].to(args.device)
        v = inputs["value"].to(args.device)
        am = inputs["atten_mask"]
        am = am.to(args.device) if isinstance(am, torch.Tensor) and am.numel() > 0 else None
        aq = _to_tensor(inputs["actual_seq_lengths_q"], args.device)
        akv = _to_tensor(inputs["actual_seq_lengths_kv"], args.device)
        bt = inputs["block_table"]
        bt = bt.to(args.device) if isinstance(bt, torch.Tensor) else None
        if sm == 3 and am is None:
            sm = 0

        print(f"\n{'─' * 128}")
        print(f"case: {case_name}  stage={meta['stage']} layer={meta['layer']}")
        print(f"  q={list(q.shape)} k={list(k.shape)} v={list(v.shape)} dtype={q.dtype}")
        print(f"  NH={NH} NKV={NKV} scale={SC:.6f} sm={sm} block_size={BS} bt={'有' if bt is not None else '无'}")
        print(f"{'─' * 128}")

        # ---------------- baseline: torch_npu 线上路径（dense） ----------------
        ms_t = _bench_one(
            lambda: run_torchnpu_faithful(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, -99.0, False),
            args.iters, args.warmup)

        # ---------------- custom: 5 lambdas × 2 stats 场景 ----------------
        results = []
        for lam in LAMBDAS:
            # 不统计
            ms_off = _bench_one(
                lambda: run_custom_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, lam,
                                      lse_flag=False, stats_flag=False),
                args.iters, args.warmup)
            # 统计（读 sparse_stats）
            _, _, ss = run_custom_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, lam,
                                     lse_flag=False, stats_flag=True)
            torch.npu.synchronize()
            ms_on = _bench_one(
                lambda: run_custom_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, lam,
                                      lse_flag=False, stats_flag=True),
                args.iters, args.warmup)
            sp, tot = parse_sparse_stats(ss)
            rate = 100.0 * sp / tot if tot > 0 else 0.0
            results.append((lam, ms_off, ms_on, sp, tot, rate))

        print(f"\n  {'lambda':>7} | {'torch_npu ms':>13} | {'custom ms (不统计)':>17} | "
              f"{'vs torch':>8} | {'custom ms (统计)':>15} | {'vs torch':>8} | "
              f"{'统计开销':>7} | {'稀疏率(统计)':>11} | {'sparse/total':>15}")
        print("  " + "-" * 122)
        for lam, ms_off, ms_on, sp, tot, rate in results:
            print(f"  {lam:>7.1f} | {ms_t:>13.4f} | {ms_off:>17.4f} | "
                  f"{ms_off / ms_t:>7.2f}x | {ms_on:>15.4f} | "
                  f"{ms_on / ms_t:>7.2f}x | {100 * (ms_on - ms_off) / ms_off:>+6.1f}% | "
                  f"{rate:>10.2f}% | {sp:>7}/{tot:<7}")

        # 输出 sanity（dense -99 不统计 vs torch_npu）
        ao_c, _, _ = run_custom_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, -99.0,
                                   lse_flag=False, stats_flag=False)
        ao_t, _ = run_torchnpu_faithful(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, -99.0, False)
        torch.npu.synchronize()
        d = (ao_c.float() - ao_t.float()).abs().nan_to_num(0)
        print(f"\n  sanity(dense -99 不统计): max_diff vs torch_npu = {d.max().item():.4e}  "
              f"NaN_c={int(torch.isnan(ao_c).sum())} NaN_t={int(torch.isnan(ao_t).sum())}")

    print("\n" + "=" * 128)
    print("完成")
    print("=" * 128)


if __name__ == "__main__":
    main()
