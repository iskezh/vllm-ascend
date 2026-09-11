#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稀疏阈值扫掠精度验证.

对 chunked_prefill dump 在多档 sparse_lambda(=BlasST threshold_log) 下对比:
  - custom 算子 vs golden(row_loop_python_blasst.optimized_blasst_sim) 的 attn_output
  - 块稀疏决策一致性: custom sparse_stats(核内统计) vs golden block_sparsity
  - 随阈值变化的稀疏度单调性

判定准则:
  - max_diff 主要看跨阈值的一致性(阈值编码 bug 会表现为个别档位 blow-up)
  - PASS  <= 0.05   (bf16 输出 vs fp32 golden 的正常舍入范围)
  - SOFT  <= 0.50   (稀疏边界块差异, 与既有 -3 档观测同量级)
  - FAIL  >  0.50

用法:
    python run_threshold_sweep.py --device npu:12
    python run_threshold_sweep.py --device npu:12 --pick 6 \
        --lambdas -99,-5,-4,-3,-2,-1,0,1,2 --scenarios sm0,sm3
    python run_threshold_sweep.py --device npu:12 --dumps a.pt b.pt
"""

import argparse, os, sys, glob

_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _CUR_DIR)

from run_dump_repro import (load_dump, run_custom_op, run_blasst_golden,   # noqa: E402
                            _parse_sparse_stats, _to_tensor)

import torch, torch_npu                                                  # noqa: E402
from vllm_ascend import platform; platform.NPUPlatform.import_kernels()  # noqa: E402
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401

DEFAULT_LAMBDAS = "-99,-5,-4,-3,-2,-1,0,1,2"
DEFAULT_DUMP_DIR = "/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps_v2"


def pick_dumps(dirpath, n):
    """从 dump 目录等距抽 n 个 chunked_prefill 用例（golden 仅支持单 seq prefill）。

    先在 CPU 上加载过滤：只留单 seq 且 q_len>1 的用例，再等距抽取。
    """
    files = sorted(glob.glob(os.path.join(dirpath, "fia_layer*_chunked_prefill*.pt")))
    if not files:
        raise FileNotFoundError(f"no chunked_prefill dumps under {dirpath}")
    single = []
    for f in files:
        try:
            aq = load_dump(f)[0]["actual_seq_lengths_q"]
            if isinstance(aq, torch.Tensor):
                if aq.dim() > 0 and aq.numel() > 1:
                    continue
                ql = int(aq.reshape(-1)[-1].item())
            else:
                ql = int(aq[-1]) if isinstance(aq, list) else int(aq)
            if ql > 1:
                single.append(f)
        except Exception:
            continue
    print(f"[sweep] dump filter: {len(files)} chunked_prefill -> {len(single)} single-seq eligible")
    if not single:
        raise FileNotFoundError("no single-seq chunked_prefill dumps")
    if len(single) <= n:
        return single
    step = len(single) / n
    return [single[min(int(i * step), len(single) - 1)] for i in range(n)]


def verdict(md):
    if md <= 0.05: return "PASS"
    if md <= 0.50: return "SOFT"
    return "FAIL"


def sweep_one(path, device, lambdas, scenarios, causal_mask):
    print(f"\n{'='*100}")
    print(f"  {os.path.basename(path)}")
    print(f"{'='*100}", flush=True)

    inputs, params, gt_attn, _, meta = load_dump(path)
    NH = params["num_heads"]; NKV = params["num_kv_heads"]
    SC = params["scale"];     BS = params["block_size"]

    q = inputs["query"].to(device); k = inputs["key"].to(device); v = inputs["value"].to(device)
    am = inputs["atten_mask"]
    if am is not None and (not isinstance(am, torch.Tensor) or am.numel() > 0):
        am = am.to(device)
    else:
        am = None
    aq = _to_tensor(inputs["actual_seq_lengths_q"], device)
    akv = _to_tensor(inputs["actual_seq_lengths_kv"], device)
    bt_raw = inputs["block_table"]
    bt = bt_raw.to(device) if bt_raw is not None else None

    if (aq.dim() > 0 and aq.numel() > 1):
        print(f"  [skip] 多 seq batch (aq numel={aq.numel()})，golden 不支持")
        return 0, 0
    q_len = int(aq.reshape(-1)[-1].item())
    if q_len <= 1:
        print(f"  [skip] decode (q_len={q_len})，golden 不支持")
        return 0, 0
    kv_len = int(akv.reshape(-1)[-1].item())
    print(f"  format={meta['format']} q_len={q_len} kv_len={kv_len} "
          f"NH={NH} NKV={NKV} D={q.shape[-1]} block_size={BS}")

    rows = []
    for tag, sm in scenarios:
        mask = causal_mask if sm == 3 else None
        if sm == 3 and am is not None:
            mask = am  # dump 自带 mask 优先
        # warmup
        _ = run_custom_op(q, k, v, mask, aq, akv, bt, NH, NKV, SC, sm, BS, lambdas[0])
        torch.npu.synchronize()

        for lam in lambdas:
            ao_c, _, ss_c = run_custom_op(q, k, v, mask, aq, akv, bt, NH, NKV, SC, sm, BS,
                                          lam, True, True)
            ao_g, blk_sp, sg_n, tot_n = run_blasst_golden(
                inputs["query"], inputs["key"], inputs["value"],
                inputs["actual_seq_lengths_q"], inputs["actual_seq_lengths_kv"],
                SC, sm, lam, device, block_table=bt_raw)
            torch.npu.synchronize()

            sc, tc = _parse_sparse_stats(ss_c)
            md = (ao_c.float().cpu() - ao_g).abs().nan_to_num(0).max().item()
            nan_c = int(torch.isnan(ao_c.float()).sum().item())
            c_ratio = sc / tc * 100 if tc > 0 else float("nan")
            g_ratio = blk_sp * 100
            rows.append((tag, sm, lam, md, nan_c, sc, tc, c_ratio, g_ratio, verdict(md)))
            print(f"  [{tag}] lam={lam:>6.1f}  max_diff(c,g)={md:.4e}  NaN={nan_c}  "
                  f"sparse c={sc}/{tc}({c_ratio:.1f}%) g={g_ratio:.1f}%  "
                  f"[{verdict(md)}]", flush=True)

    # 汇总表
    print(f"\n  {'─'*96}")
    print(f"  {'scenario':<8} {'lambda':>7} {'max_diff':>12} {'NaN':>6} "
          f"{'c_sparse%':>10} {'g_sparse%':>10} {'Δsp%':>7} {'verdict':>7}")
    print(f"  {'─'*96}")
    fails = 0
    for tag, sm, lam, md, nan_c, sc, tc, c_ratio, g_ratio, vd in rows:
        d_sp = (c_ratio - g_ratio) if (tc > 0) else float("nan")
        print(f"  {tag:<8} {lam:>7.1f} {md:>12.4e} {nan_c:>6} "
              f"{c_ratio:>10.1f} {g_ratio:>10.1f} {d_sp:>7.1f} {vd:>7}")
        if vd == "FAIL":
            fails += 1
    print(f"  {'─'*96}")
    print(f"  TOTAL rows={len(rows)}  FAIL={fails}", flush=True)
    return fails, len(rows)


def main():
    p = argparse.ArgumentParser(description="稀疏阈值扫掠: custom vs golden")
    p.add_argument("--dump-dir", default=DEFAULT_DUMP_DIR)
    p.add_argument("--dumps", nargs="*", default=None, help="指定 dump 文件（覆盖 --pick）")
    p.add_argument("--pick", type=int, default=6, help="从 dump 目录等距抽取的用例数")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--lambdas", default=DEFAULT_LAMBDAS,
                   help="逗号分隔的 sparse_lambda(=threshold_log) 列表")
    p.add_argument("--scenarios", default="sm0,sm3",
                   help="逗号分隔: sm0(无mask) sm3(causal)")
    args = p.parse_args()

    device = args.device
    torch.npu.set_device(device)
    lambdas = [float(x) for x in args.lambdas.split(",")]
    scenarios = [(s.strip(), int(s.strip()[2:])) for s in args.scenarios.split(",")]
    dumps = args.dumps or pick_dumps(args.dump_dir, args.pick)

    print(f"[sweep] device={device} dumps={len(dumps)} lambdas={lambdas} scenarios={scenarios}")

    causal_mask = torch.triu(torch.ones(2048, 2048, device=device, dtype=torch.bool),
                             diagonal=1)
    tot_fail = tot_row = 0
    for i, path in enumerate(dumps):
        print(f"\n[sweep] ===== dump {i+1}/{len(dumps)} =====", flush=True)
        f, n = sweep_one(path, device, lambdas, scenarios, causal_mask)
        tot_fail += f; tot_row += n

    print(f"\n{'#'*100}")
    print(f"[sweep] ALL DONE: rows={tot_row} FAIL={tot_fail} "
          f"({'ALL PASS' if tot_fail == 0 else 'HAS FAILURES'})")
    print(f"{'#'*100}")


if __name__ == "__main__":
    main()
