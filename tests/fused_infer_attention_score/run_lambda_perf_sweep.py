#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单算子 λ 性能扫掠: baseline(torch_npu)@-99 为 dense 基准, custom 在 λ∈{-99,-9..-1} 各档计时,
关联稀疏率(sparse_stats), 覆盖 chunked_prefill + decode 两形态.
用法: python run_lambda_perf_sweep.py --device npu:0
"""
import argparse, os, sys

_CUR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _CUR)
from run_dump_repro import (load_dump, run_custom_op, run_torchnpu_op,   # noqa: E402
                            _parse_sparse_stats, _to_tensor, _bench_one)

import torch, torch_npu                                                  # noqa: E402
from vllm_ascend import platform; platform.NPUPlatform.import_kernels()  # noqa: E402
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401

DUMPS = [
    ("chunked_prefill", "/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps_v2/fia_layer00_rank0_chunked_prefill_occ1.pt"),
    ("decode",          "/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps_v2/fia_layer00_rank0_decode_occ1.pt"),
]
LAMS_CUSTOM = [-99.0] + [float(x) for x in range(-9, 0)]   # -99, -9..-1
LAMS_BASE = [-99.0]
ITERS, WARMUP = 100, 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:0")
    args = ap.parse_args()
    torch.npu.set_device(args.device)

    for tag, path in DUMPS:
        inputs, params, _, _, meta = load_dump(path)
        NH = params["num_heads"]; NKV = params["num_kv_heads"]
        SC = params["scale"]; BS = params["block_size"]
        sm = int(params.get("sparse_mode", 3))
        if sm == 3:
            sm = 3  # 两类 dump 均带 mask 场景, 保持 dump 原样
        q = inputs["query"].to(args.device); k = inputs["key"].to(args.device); v = inputs["value"].to(args.device)
        am = inputs["atten_mask"]
        am = am.to(args.device) if am is not None and am.numel() > 0 else None
        aq = _to_tensor(inputs["actual_seq_lengths_q"], args.device)
        akv = _to_tensor(inputs["actual_seq_lengths_kv"], args.device)
        bt = inputs["block_table"]
        bt = bt.to(args.device) if bt is not None else None

        print(f"\n{'='*78}")
        print(f"  [{tag}] q={list(q.shape)} kv_heads={NKV} sm={sm} iters={ITERS}")
        print(f"{'='*78}")
        print(f"  {'impl':<10} {'lambda':>6} {'ms/call':>9} {'vs base@-99':>12} {'sparse%':>8}")

        base_ms = None
        rows = []
        for impl in ("base", "custom"):
            for lam in (LAMS_BASE if impl == "base" else LAMS_CUSTOM):
                if impl == "base":
                    ms = _bench_one(lambda: run_torchnpu_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, lam),
                                    ITERS, WARMUP)
                    ss = None
                else:
                    ms = _bench_one(lambda: run_custom_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, lam),
                                    ITERS, WARMUP)
                    _, _, ss = run_custom_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, lam, False, True)
                    ss = _parse_sparse_stats(ss)
                if impl == "base" and lam == -99.0:
                    base_ms = ms
                ratio = ms / base_ms if base_ms else float("nan")
                sp = f"{ss[0]/ss[1]*100:.1f}" if ss and ss[1] > 0 else "—"
                rows.append((impl, lam, ms, ratio, sp))
                print(f"  {impl:<10} {lam:>6.1f} {ms:>9.4f} {ratio:>11.3f}x {sp:>8}", flush=True)
        # 汇总
        print(f"  {'─'*78}")
        c99 = [r for r in rows if r[0] == "custom" and r[1] == -99.0][0]
        best = min((r for r in rows if r[0] == "custom"), key=lambda r: r[2])
        print(f"  dense 开销: custom@-99 / baseline@-99 = {c99[3]:.3f}x")
        print(f"  最快档: custom@{best[1]:.0f} = {best[2]:.4f} ms ({best[3]:.3f}x, sparse {best[4]}%)")


if __name__ == "__main__":
    main()
