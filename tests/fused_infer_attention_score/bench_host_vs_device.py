#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分解 custom vs baseline 单算子耗时：host wall（异步发射）vs event（含 host 间隙）。

判定逻辑:
  - host_wall ≈ total   → host-bound（tiling/executor/H2D 占主导）
  - host_wall << total  → device kernel-bound
"""
import os, sys, time

_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
_CUSTOM_OPP_PATH = os.path.join(
    _CUR_DIR, "..", "..", "vllm_ascend", "_cann_ops_custom", "vendors", "vllm-ascend")
if os.path.exists(_CUSTOM_OPP_PATH):
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = _CUSTOM_OPP_PATH

import torch, torch_npu
from vllm_ascend import platform; platform.NPUPlatform.import_kernels()
import vllm_ascend.vllm_ascend_C  # noqa

sys.path.insert(0, _CUR_DIR)
from run_dump_repro import load_dump, run_custom_op, run_torchnpu_faithful, SWA_INT_MAX

DUMPS = {
    "decode": "/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps_v2/fia_layer00_rank0_decode_occ1.pt",
    "chunked": "/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps_v2/fia_layer00_rank0_chunked_prefill_occ1.pt",
}


def bench_split(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    # host wall（纯异步发射耗时，不含 device 排空）
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    host_wall = (time.perf_counter() - t0) / iters * 1e3
    torch.npu.synchronize()
    # event 时间（stream 上两点间 wall，含 host 间隙）
    e0 = torch.npu.Event(enable_timing=True); e1 = torch.npu.Event(enable_timing=True)
    torch.npu.synchronize()
    e0.record()
    for _ in range(iters):
        fn()
    e1.record(); torch.npu.synchronize()
    event_ms = e0.elapsed_time(e1) / iters
    return host_wall, event_ms


def main():
    device = "npu:0"
    torch.npu.set_device(device)
    for name, path in DUMPS.items():
        inputs, params, gt, gt_lse, meta = load_dump(path)
        NH, NKV, SC, BS = params["num_heads"], params["num_kv_heads"], params["scale"], params["block_size"]
        lam = float(params.get("sparse_lambda", -99.0))
        sm = int(params.get("sparse_mode", 3))
        lse = bool(params.get("enable_lse_flag", False))
        q = inputs["query"].to(device); k = inputs["key"].to(device); v = inputs["value"].to(device)
        am0 = inputs["atten_mask"]
        am = am0.to(device) if am0 is not None and isinstance(am0, torch.Tensor) and am0.numel() > 0 else None
        aq = inputs["actual_seq_lengths_q"]
        akv = inputs["actual_seq_lengths_kv"]
        aq = aq.to(device) if isinstance(aq, torch.Tensor) else torch.tensor(aq, device=device)
        akv = akv.to(device) if isinstance(akv, torch.Tensor) else torch.tensor(akv, device=device)
        bt = inputs["block_table"]
        bt = bt.to(device) if bt is not None else None
        if sm == 3 and am is None:
            sm = 0

        print(f"\n=== {name}: q={list(q.shape)} sm={sm} lam={lam} ===")
        hw_c, ev_c = bench_split(lambda: run_custom_op(q, k, v, am, aq, akv, bt,
                                                       NH, NKV, SC, sm, BS, lam, lse))
        hw_b, ev_b = bench_split(lambda: run_torchnpu_faithful(q, k, v, am, aq, akv, bt,
                                                               NH, NKV, SC, sm, BS, lam, lse))
        print(f"  custom:   host_wall={hw_c:8.3f} ms   event={ev_c:8.3f} ms")
        print(f"  baseline: host_wall={hw_b:8.3f} ms   event={ev_b:8.3f} ms")
        for tag, hw, ev in (("custom", hw_c, ev_c), ("baseline", hw_b, ev_b)):
            bound = "HOST-bound (tiling/executor/H2D)" if hw > ev * 0.5 else "DEVICE-kernel-bound"
            print(f"  {tag:<9} -> {bound}")


if __name__ == "__main__":
    main()
