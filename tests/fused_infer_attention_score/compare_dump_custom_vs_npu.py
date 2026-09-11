#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比 dump 输入下 custom FIA(λ=-99) 与 torch_npu 基线 FIA 的精度。

输入: _fia_repro_dump 产出的格式A dump (attention_v1.py forward_custom_fused_infer_attention)。
对每份 dump:
  1. custom  : torch.ops._C_ascend.npu_fused_infer_attention_score, 参数与 serving 完全一致
               (int64 device seq lens, λ 取 dump 中 params.sparse_lambda)
  2. base    : torch_npu.npu_fused_infer_attention_score, 与 baseline 分支
               (attention_v1.py forward_fused_infer_attention :1263) 调用形式一致
               (host list seq lens, 无 antiquant_mode, sparse_mode=3)
  3. sanity  : custom 重放 vs dump 时保存的 serving 现场输出

用法:
    python compare_dump_custom_vs_npu.py --dump /path/to/file.pt
    python compare_dump_custom_vs_npu.py --dir  /path/to/attention_dumps_v2
"""

import argparse
import glob
import os
import sys

_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
_CUSTOM_OPP_PATH = os.path.join(
    _CUR_DIR, "..", "..", "vllm_ascend", "_cann_ops_custom", "vendors",
    "vllm-ascend")
if os.path.exists(_CUSTOM_OPP_PATH):
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = _CUSTOM_OPP_PATH

import torch
import torch_npu  # noqa: F401

from vllm_ascend import platform

platform.NPUPlatform.import_kernels()
import vllm_ascend.vllm_ascend_C  # noqa: F401

SWA_INT_MAX = 2147483647


def _progress(msg):
    print(msg, flush=True)


def run_custom(inp, params, device):
    """与 serving custom 分支完全一致的调用。"""
    q = inp["query"].to(device)
    k = inp["key"].to(device)
    v = inp["value"].to(device)
    am = inp["atten_mask"]
    am = am.to(device) if isinstance(am, torch.Tensor) and am.numel() > 0 else None
    bt = inp["block_table"]
    bt = bt.to(device) if isinstance(bt, torch.Tensor) and bt.numel() > 0 else None
    aq = inp["actual_seq_lengths_q"].cpu().tolist()
    akv = inp["actual_seq_lengths_kv"].cpu().tolist()
    out, lse, _ = torch.ops._C_ascend.npu_fused_infer_attention_score(
        q, k, v, None, am, aq, akv, bt,
        params["num_heads"], params["scale"],
        params.get("pre_tokens", SWA_INT_MAX),
        params.get("next_tokens", SWA_INT_MAX),
        "TND", params["num_kv_heads"], params["sparse_mode"],
        params.get("inner_precise", 0), params["block_size"], 0,
        params["sparse_lambda"], params.get("enable_lse_flag", False))
    return out


def run_base(inp, params, device):
    """与 baseline 分支 (forward_fused_infer_attention 非 sinks 路径) 一致的调用。"""
    q = inp["query"].to(device)
    k = inp["key"].to(device)
    v = inp["value"].to(device)
    am = inp["atten_mask"]
    am = am.to(device) if isinstance(am, torch.Tensor) and am.numel() > 0 else None
    bt = inp["block_table"]
    bt = bt.to(device) if isinstance(bt, torch.Tensor) and bt.numel() > 0 else None
    # baseline 传 host list
    aq = inp["actual_seq_lengths_q"].cpu().tolist()
    akv = inp["actual_seq_lengths_kv"].cpu().tolist()
    out, _ = torch_npu.npu_fused_infer_attention_score(
        query=q, key=k, value=v, atten_mask=am, block_table=bt,
        input_layout="TND", block_size=params["block_size"],
        actual_seq_lengths=aq, actual_seq_lengths_kv=akv,
        num_key_value_heads=params["num_kv_heads"],
        num_heads=params["num_heads"], scale=params["scale"],
        sparse_mode=params["sparse_mode"])
    return out


def diff_stats(a, b, label):
    a32 = a.detach().float().cpu()
    b32 = b.detach().float().cpu()
    if a32.shape != b32.shape:
        return {"label": label, "shape_mismatch": (tuple(a32.shape), tuple(b32.shape))}
    d = (a32 - b32).abs()
    stats = {
        "label": label,
        "max": d.max().item(),
        "mean": d.mean().item(),
        "n_gt_1e-2": int((d > 1e-2).sum().item()),
        "numel": d.numel(),
    }
    # 定位最大 diff 的 (token, head)
    idx = d.flatten().argmax().item()
    if d.dim() == 3:
        t, h, c = torch.unravel_index(torch.tensor(idx), d.shape)
        stats["argmax_pos"] = (int(t), int(h), int(c))
        stats["per_head_max"] = d.amax(dim=(0, 2)).tolist()
    return stats


def compare_one(path, device):
    data = torch.load(path, map_location="cpu", weights_only=False)
    inp, params = data["inputs"], data["params"]
    gt = data.get("outputs", {}).get("attn_output")
    stage = data.get("stage", "?")
    layer = data.get("layer_name", "?")
    rank = data.get("tp_rank", "?")

    _progress(f"\n{'=' * 80}")
    _progress(f"{os.path.basename(path)}")
    _progress(f"  layer={layer} rank={rank} stage={stage} "
              f"num_tokens={data.get('num_tokens')} "
              f"lambda={params['sparse_lambda']}")
    _progress(f"  q={tuple(inp['query'].shape)} k={tuple(inp['key'].shape)} "
              f"bt={None if inp['block_table'] is None else tuple(inp['block_table'].shape)} "
              f"mask={None if inp['atten_mask'] is None else tuple(inp['atten_mask'].shape)}")
    _progress(f"  aq={inp['actual_seq_lengths_q'].tolist()[:8]}{'...' if inp['actual_seq_lengths_q'].numel() > 8 else ''} "
              f"akv={inp['actual_seq_lengths_kv'].tolist()[:8]}{'...' if inp['actual_seq_lengths_kv'].numel() > 8 else ''}")

    with torch.npu.device(device):
        out_c = run_custom(inp, params, device)
        torch.npu.synchronize()
        out_n = run_base(inp, params, device)
        torch.npu.synchronize()

    rows = [diff_stats(out_c, out_n, "custom vs base")]
    if gt is not None:
        rows.append(diff_stats(out_c, gt, "custom vs serving-dump"))
        rows.append(diff_stats(out_n, gt, "base   vs serving-dump"))

    for r in rows:
        if "shape_mismatch" in r:
            _progress(f"  [{r['label']}] SHAPE MISMATCH {r['shape_mismatch']}")
            continue
        _progress(f"  [{r['label']}] max={r['max']:.6e} mean={r['mean']:.6e} "
                  f">1e-2: {r['n_gt_1e-2']}/{r['numel']} argmax(t,h,c)={r.get('argmax_pos')}")
        if r.get("per_head_max"):
            ph = " ".join(f"h{i}={v:.3e}" for i, v in enumerate(r["per_head_max"]))
            _progress(f"      per-head max: {ph}")
    return rows[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", default=None)
    p.add_argument("--dir", default=None)
    p.add_argument("--device", default="npu:0")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.npu.set_device(device)

    files = []
    if args.dump:
        files = [args.dump]
    elif args.dir:
        files = sorted(glob.glob(os.path.join(args.dir, "*.pt")))
    if not files:
        _progress("no dump files found")
        sys.exit(1)

    summary = []
    for f in files:
        try:
            r = compare_one(f, device)
            r["file"] = os.path.basename(f)
            summary.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            summary.append({"file": os.path.basename(f), "error": str(e)})

    _progress(f"\n{'=' * 100}")
    _progress(f"{'file':<55} {'max_diff':>12} {'mean_diff':>12} {'>1e-2':>12}")
    _progress("-" * 100)
    for r in sorted(summary, key=lambda x: -x.get("max", 0)):
        if "error" in r:
            _progress(f"{r['file']:<55} ERROR: {r['error']}")
        elif "shape_mismatch" in r:
            _progress(f"{r['file']:<55} SHAPE MISMATCH {r['shape_mismatch']}")
        else:
            _progress(f"{r['file']:<55} {r['max']:>12.6e} {r['mean']:>12.6e} "
                      f"{r['n_gt_1e-2']:>6}/{r['numel']}")
    _progress("=" * 100)


if __name__ == "__main__":
    main()
