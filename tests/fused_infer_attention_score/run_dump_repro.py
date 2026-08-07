#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单算子复现 & 多维度对比脚本.

支持三种 dump 格式:
  格式A: 我们的 dump, data["params"] + data["outputs"]["attn_output","softmax_lse"]
  格式B: FIA dump,    data["inputs"] + data["output"] (无 lse)
  格式C: ops-transformer-dev dump, data["inputs"] (scalar tensors) + data["outputs"]["attn_output"]

对比维度 (4 scenarios × custom/npu/golden):
  - sparse/total 稀疏度
  - custom vs npu    max_diff
  - custom vs golden max_diff
  - npu vs golden    max_diff

Golden: row_loop_python_blasst.optimized_blasst_sim (npu_head_mode=True)

用法:
    python run_dump_repro.py --dump /path/to/dump.pt
    python run_dump_repro.py --dump /path/to/dump.pt --device npu:2
"""

import argparse, os, sys

_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
_CUSTOM_OPP_PATH = os.path.join(
    _CUR_DIR, "..", "..", "vllm_ascend", "_cann_ops_custom", "vendors", "vllm-ascend")
if os.path.exists(_CUSTOM_OPP_PATH):
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = _CUSTOM_OPP_PATH

# Add ops-transformer-dev tests path for row_loop_python_blasst
_OPS_TRANSFORMER_TESTS = (
    "/home/z00603376/ops-transformer-dev/attention/fused_infer_attention_score/tests/tests")
if _OPS_TRANSFORMER_TESTS not in sys.path:
    sys.path.insert(0, _OPS_TRANSFORMER_TESTS)

import torch, torch_npu
from vllm_ascend import platform; platform.NPUPlatform.import_kernels()
import vllm_ascend.vllm_ascend_C  # noqa

from row_loop_python_blasst import optimized_blasst_sim, GlobalConfig as BlasstGlobalConfig

SWA_INT_MAX = 2147483647


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════
def _progress(msg): print(msg, flush=True)

def _parse_lse_per_core(lse, max_cores=24):
    data_1d = lse.flatten().cpu().float()
    cores = min(max_cores, len(data_1d) // 16)
    per_core = []; ts = tb = 0
    for i in range(cores):
        s = int(round(data_1d[i * 16].item()))
        b = int(round(data_1d[i * 16 + 1].item()))
        ts += s; tb += b; per_core.append((i, s, b))
    return ts, tb, per_core


# ═══════════════════════════════════════════════════════════════
# Dump loader — detects format A/B/C
# ═══════════════════════════════════════════════════════════════
def load_dump(path):
    data = torch.load(path, map_location="cpu", weights_only=False)

    if "params" in data and "outputs" in data:
        # Format A
        inp, params, gt = data["inputs"], data["params"], data["outputs"]["attn_output"]
        gt_lse = data["outputs"].get("softmax_lse")
        meta = {"format": "A", "stage": data.get("stage", "?"), "layer": data.get("layer_name", "?"),
                "has_lse": gt_lse is not None}
    elif "output" in data:
        # Format B
        raw = data["inputs"]
        params = {"num_heads": raw["num_heads"], "num_kv_heads": raw["num_key_value_heads"],
                  "scale": raw["scale"], "sparse_mode": raw["sparse_mode"], "block_size": raw["block_size"],
                  "enable_lse_flag": True}
        inp = {"query": raw["query"], "key": raw["key"], "value": raw["value"],
               "atten_mask": raw["atten_mask"], "block_table": raw["block_table"],
               "actual_seq_lengths_q": raw["actual_seq_lengths"],
               "actual_seq_lengths_kv": raw["actual_seq_lengths_kv"]}
        gt = data["output"]; gt_lse = None
        params["sparse_lambda"] = (float(raw["antiquant_mode"]) + 100.0) / 10.0
        meta = {"format": "B", "stage": data.get("attn_state", "?"),
                "layer": f"layer_{data.get('layer_idx','?')}",
                "antiquant_mode": raw["antiquant_mode"], "has_lse": False}
    elif "layer_id" in data and "outputs" in data:
        # Format C: ops-transformer-dev dump
        raw = data["inputs"]
        def _ival(t): return int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        def _fval(t): return float(t.item()) if isinstance(t, torch.Tensor) else float(t)
        params = {"num_heads": _ival(raw["num_heads"]), "num_kv_heads": _ival(raw["num_kv_heads"]),
                  "scale": _fval(raw["scale"]), "block_size": _ival(raw["block_size"]),
                  "enable_lse_flag": True}
        bt = raw.get("block_table")
        if bt is not None and (isinstance(bt, torch.Tensor) and bt.numel() == 0):
            bt = None
        inp = {"query": raw["query"], "key": raw["key"], "value": raw["value"],
               "atten_mask": None, "block_table": bt,
               "actual_seq_lengths_q": raw["actual_seq_qlen"],
               "actual_seq_lengths_kv": raw["actual_seq_kvlen"]}
        gt = data["outputs"]["attn_output"]; gt_lse = None
        meta = {"format": "C", "layer": f"layer_{data['layer_id']}", "has_lse": False}
    else:
        raise ValueError(f"Unknown dump format: keys={list(data.keys())}")

    return inp, params, gt, gt_lse, meta


# ═══════════════════════════════════════════════════════════════
# Op wrappers
# ═══════════════════════════════════════════════════════════════
def _to_tensor(x, device, dtype=torch.int64):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return torch.tensor(x, dtype=dtype, device=device)
    else:
        return torch.tensor([x], dtype=dtype, device=device)

def run_custom_op(query, key, value, atten_mask, aq, akv, bt, num_heads, num_kv_heads,
                  scale, sparse_mode, block_size, sparse_lam, lse_flag=True):
    return torch.ops._C_ascend.npu_fused_infer_attention_score(
        query, key, value, None, atten_mask, aq, akv, bt,
        num_heads, scale, SWA_INT_MAX, SWA_INT_MAX, "TND",
        num_kv_heads, sparse_mode, 0, block_size, 0, sparse_lam, lse_flag)

def run_torchnpu_op(query, key, value, atten_mask, aq, akv, bt, num_heads, num_kv_heads,
                    scale, sparse_mode, block_size, sparse_lam, lse_flag=True):
    aq_mode = int(sparse_lam * 10 - 100)
    return torch_npu.npu_fused_infer_attention_score(
        query=query, key=key, value=value, atten_mask=atten_mask,
        block_table=bt, input_layout="TND", block_size=block_size,
        actual_seq_lengths=aq, actual_seq_lengths_kv=akv,
        num_key_value_heads=num_kv_heads, num_heads=num_heads,
        scale=scale, sparse_mode=sparse_mode,
        antiquant_mode=aq_mode, softmax_lse_flag=lse_flag)


def run_torchnpu_faithful(query, key, value, atten_mask, aq, akv, bt, num_heads,
                          num_kv_heads, scale, sparse_mode, block_size, sparse_lam,
                          lse_flag=False):
    """镜像线上 baseline 调用（attention_v1.forward_fused_infer_attention）：
    dense(-99) 时线上不传 antiquant_mode（标准 dense 调用），sparse 时才走
    BlasST antiquant_mode 编码。
    """
    if sparse_lam <= -99.0:
        return torch_npu.npu_fused_infer_attention_score(
            query=query, key=key, value=value, atten_mask=atten_mask,
            block_table=bt, input_layout="TND", block_size=block_size,
            actual_seq_lengths=aq, actual_seq_lengths_kv=akv,
            num_key_value_heads=num_kv_heads, num_heads=num_heads,
            scale=scale, sparse_mode=sparse_mode,
            softmax_lse_flag=lse_flag)
    return run_torchnpu_op(query, key, value, atten_mask, aq, akv, bt, num_heads,
                           num_kv_heads, scale, sparse_mode, block_size, sparse_lam,
                           lse_flag)


# ═══════════════════════════════════════════════════════════════
# BLASST Golden (row_loop_python_blasst, npu_head_mode=True)
# ═══════════════════════════════════════════════════════════════
def run_blasst_golden(query, key, value, aq, akv, scale, sparse_mode, threshold_log, device):
    """Run optimized_blasst_sim as golden reference (on specified device).

    Returns (attn_output, block_sparsity, sparse_block_count, total_block_count).
    attn_output is FP32 on CPU.
    """
    # optimized_blasst_sim expects 0-dim tensors for lengths
    if isinstance(aq, torch.Tensor):
        if aq.dim() == 0: aq_s = aq
        elif aq.numel() == 1: aq_s = aq.reshape(())
        else: aq_s = aq[-1].reshape(())
    else:
        v = aq[-1] if isinstance(aq, list) else aq
        aq_s = torch.tensor(v)

    if isinstance(akv, torch.Tensor):
        if akv.dim() == 0: akv_s = akv
        elif akv.numel() == 1: akv_s = akv.reshape(())
        else: akv_s = akv[-1].reshape(())
    else:
        v = akv[-1] if isinstance(akv, list) else akv
        akv_s = torch.tensor(v)

    out, row_sp, block_sp = optimized_blasst_sim(
        query.to(device), key.to(device), value.to(device),
        aq_s.to(device), akv_s.to(device), scale,
        threshold_log=threshold_log, sparse_mode=sparse_mode,
        npu_head_mode=True, print_flag=False)

    out_cpu = out.cpu().float()
    torch.npu.synchronize()

    # Compute approximate sparse/total block counts
    QS = BlasstGlobalConfig.QS_BLOCK    # 128
    KV = BlasstGlobalConfig.KV_BLOCK    # 512
    q_len = int(aq_s.item()); kv_len = int(akv_s.item())
    total_block = math.ceil(q_len / QS) * math.ceil(kv_len / KV) * query.size(1)  # * num_heads
    if block_sp > 0:
        sparse_block = int(round(block_sp * total_block))
    else:
        sparse_block = 0

    return out_cpu, block_sp, sparse_block, total_block


# ═══════════════════════════════════════════════════════════════
# Main test
# ═══════════════════════════════════════════════════════════════
import math

def run_one_case(path, device):
    _progress(f"\n{'='*80}")
    _progress(f"  {os.path.basename(path)}")
    _progress(f"{'='*80}")

    inputs, params, gt_attn, gt_lse, meta = load_dump(path)
    _progress(f"  format={meta['format']} layer={meta['layer']}")

    NH  = params["num_heads"]; NKV = params["num_kv_heads"]
    SC  = params["scale"];     BS  = params["block_size"]

    # Move to device
    q  = inputs["query"].to(device); k = inputs["key"].to(device); v = inputs["value"].to(device)
    am_orig = inputs["atten_mask"]
    am = am_orig.to(device) if am_orig is not None and (not isinstance(am_orig, torch.Tensor) or am_orig.numel() > 0) else None
    aq = _to_tensor(inputs["actual_seq_lengths_q"], device)
    akv= _to_tensor(inputs["actual_seq_lengths_kv"], device)
    bt_raw = inputs["block_table"]
    bt = bt_raw.to(device) if bt_raw is not None else None

    _progress(f"  q={list(q.shape)} k={list(k.shape)} v={list(v.shape)}")
    _progress(f"  NH={NH} NKV={NKV} scale={SC:.6f} block_size={BS}")
    _progress(f"  aq_dim={aq.dim()}, aq={aq.cpu().tolist() if aq.dim()>0 else aq.item()}")

    # ═══════════════════════════════════════════════════════════
    # AS-DUMPED 保真回放：用 dump 的原始参数/输入跑 custom vs npu
    # （raw dump 含整池 KV + 原始 block_table，保留陈旧列等越读现场）
    # ═══════════════════════════════════════════════════════════
    lam0 = float(params.get("sparse_lambda", -99.0))
    sm0 = int(params.get("sparse_mode", 3))
    lse0 = bool(params.get("enable_lse_flag", False))
    if sm0 == 3 and am is None:
        # sm=3 需要 mask；无 mask 时该组合非法（custom kernel 会 MTE 越界），降级 sm=0
        _progress(f"  [skip] sm=3 + mask=None 非法组合，降级为 sm=0")
        sm0 = 0
    _progress(f"\n  --- AS-DUMPED 保真回放: sm={sm0} lambda={lam0} lse={lse0} "
              f"mask={'dumped' if am is not None else None} ---")
    ao_c0, _ = run_custom_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm0, BS, lam0, lse0)
    ao_n0, _ = run_torchnpu_faithful(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm0, BS, lam0, lse0)
    torch.npu.synchronize()
    nc0 = int(torch.isnan(ao_c0).sum().item())
    nn0 = int(torch.isnan(ao_n0).sum().item())
    d0 = (ao_c0.float() - ao_n0.float()).abs().nan_to_num(0)
    if ao_c0.dtype in (torch.bfloat16, torch.float16):
        bit0 = int((ao_c0.view(torch.int16) != ao_n0.view(torch.int16)).sum().item())
    else:
        bit0 = -1
    print(f"      custom NaN={nc0}/{ao_c0.numel()} ({nc0/ao_c0.numel()*100:.2f}%)  "
          f"npu NaN={nn0}/{ao_n0.numel()}")
    print(f"      max_diff(有限)={d0.max().item():.6e}  bit_diff_elems={bit0}")
    if isinstance(gt_attn, torch.Tensor):
        gn0 = int(torch.isnan(gt_attn).sum().item())
        dg0 = (ao_c0.float().cpu() - gt_attn.float()).abs().nan_to_num(0)
        print(f"      vs 线上输出: 线上 NaN={gn0}/{gt_attn.numel()}  "
              f"max_diff={dg0.max().item():.6e}")

    # 多 seq batch：golden 仅支持单 seq，4-scenario 矩阵跳过
    single_seq = (aq.dim() == 0) or (aq.numel() == 1)
    if not single_seq:
        _progress(f"\n  多 seq batch (aq numel={aq.numel()})："
                  f"跳过 golden/4-scenario 矩阵，保真回放结果即结论。")
        return
    q_len0 = int(aq.reshape(-1)[-1].item())
    if q_len0 <= 1:
        _progress(f"\n  decode(q_len={q_len0})：2048² mask 与单 token query 不匹配，"
                  f"跳过 golden/4-scenario 矩阵，保真回放结果即结论。")
        return

    # Build causal mask for sm=3 scenario
    causal_mask = torch.triu(torch.ones(2048, 2048, device=device, dtype=torch.bool), diagonal=1)

    # Warmup
    _progress("  Warming up ...")
    for sm, mask in [(0, None), (3, causal_mask)]:
        for lam in [-99.0, -3.0]:
            _ = run_torchnpu_op(q, k, v, mask, aq, akv, bt, NH, NKV, SC, sm, BS, lam)
            _ = run_custom_op(q, k, v, mask, aq, akv, bt, NH, NKV, SC, sm, BS, lam)
    torch.npu.synchronize()

    # ═══════════════════════════════════════════════════════════
    # 4 scenarios + golden
    # ═══════════════════════════════════════════════════════════
    results = {}   # (label, op_name) -> (ao, lse, ms)
    golden_res = {} # label -> (ao_g, blk_sp, sparse_n, total_n)
    scenarios = [
        ("sm=0 mask=None  dense(-99)", 0, None, -99.0),
        ("sm=0 mask=None  sparse(-3)", 0, None, -3.0),
        ("sm=3 mask=2048² dense(-99)", 3, causal_mask, -99.0),
        ("sm=3 mask=2048² sparse(-3)", 3, causal_mask, -3.0),
    ]

    _progress(f"\n  --- Golden (row_loop_python_blasst, npu_head_mode=True) ---")
    golden_res = {}
    for idx, (label, sm, mask, lam) in enumerate(scenarios):
        _progress(f"    [{idx+1}/4] {label} ...")
        ao_g, blk_sp, sg_n, tot_n = run_blasst_golden(
            inputs["query"], inputs["key"], inputs["value"],
            inputs["actual_seq_lengths_q"], inputs["actual_seq_lengths_kv"],
            SC, sm, lam, device)
        golden_res[label] = (ao_g, blk_sp, sg_n, tot_n)
        _progress(f"         block_sparsity={blk_sp*100:.2f}%")

    for label, sm, mask, lam in scenarios:
        _progress(f"\n  [{label}]")
        # custom
        t0 = torch.npu.Event(enable_timing=True); t1 = torch.npu.Event(enable_timing=True)
        t0.record()
        ao_c, lse_c = run_custom_op(q, k, v, mask, aq, akv, bt, NH, NKV, SC, sm, BS, lam)
        t1.record(); torch.npu.synchronize()
        results[(label, "custom")] = (ao_c, lse_c, t0.elapsed_time(t1))
        # npu
        t2 = torch.npu.Event(enable_timing=True); t3 = torch.npu.Event(enable_timing=True)
        t2.record()
        ao_n, lse_n = run_torchnpu_op(q, k, v, mask, aq, akv, bt, NH, NKV, SC, sm, BS, lam)
        t3.record(); torch.npu.synchronize()
        results[(label, "npu")] = (ao_n, lse_n, t2.elapsed_time(t3))

        sc, tc, _ = _parse_lse_per_core(lse_c)
        sn, tn, _ = _parse_lse_per_core(lse_n)
        xd = (ao_c.float() - ao_n.float()).abs()
        x_ok = "bit-exact" if xd.max().item() == 0 else str(torch.allclose(ao_c.float(), ao_n.float(), rtol=1e-2, atol=1e-3))

        # Golden diffs
        ao_g = golden_res[label][0]
        gd_c = (ao_g - ao_c.float().cpu()).abs().max().item()
        gd_n = (ao_g - ao_n.float().cpu()).abs().max().item()

        print(f"      custom sparse={sc}/{tc} ({sc/tc*100:.1f}%)   npu sparse={sn}/{tn} ({sn/tn*100:.1f}%)" if tc>0 and tn>0 else f"      N/A")
        print(f"      golden sparse={golden_res[label][2]}/{golden_res[label][3]} ({golden_res[label][1]*100:.1f}%)")
        print(f"      custom vs npu:     max_diff={xd.max().item():.6e}  allclose={x_ok}")
        print(f"      custom vs golden:  max_diff={gd_c:.6e}")
        print(f"      npu vs golden:     max_diff={gd_n:.6e}")

    # ═══════════════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════════════
    print(f"\n  {'─'*120}")
    print(f"  {'Scenario':<32} {'custom':>12} {'npu':>12} {'golden':>12} {'c vs n':>10} {'c vs g':>10} {'n vs g':>10}")
    print(f"  {'':>32} {'sparse/total':>12} {'sparse/total':>12} {'sparse/total':>12} {'max_diff':>10} {'max_diff':>10} {'max_diff':>10}")
    print(f"  {'─'*120}")

    for label, sm, mask, lam in scenarios:
        ao_c, lse_c, _ = results[(label, "custom")]
        ao_n, lse_n, _ = results[(label, "npu")]
        sc, tc, _ = _parse_lse_per_core(lse_c)
        sn, tn, _ = _parse_lse_per_core(lse_n)
        ao_g, blk_sp, sg_n, tot_n = golden_res[label]

        sc_str = f"{sc}/{tc}" if tc > 0 else "N/A"
        sn_str = f"{sn}/{tn}" if tn > 0 else "N/A"
        sg_str = f"{sg_n}/{tot_n}" if tot_n > 0 else "N/A"
        xd = (ao_c.float() - ao_n.float()).abs().max().item()
        gd_c = (ao_g - ao_c.float().cpu()).abs().max().item()
        gd_n = (ao_g - ao_n.float().cpu()).abs().max().item()

        print(f"  {label:<32} {sc_str:>12} {sn_str:>12} {sg_str:>12} {xd:>10.6e} {gd_c:>10.6e} {gd_n:>10.6e}")
    print(f"  {'─'*120}")


# ═══════════════════════════════════════════════════════════════
# Entry
# ═══════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="单算子复现 & multi-scenario 对比 (golden=blasst_sim)")
    p.add_argument("--dump", default=None, help=".pt dump 文件路径")
    p.add_argument("--device", default="npu:0")
    args = p.parse_args()

    device = args.device; torch.npu.set_device(device)

    if args.dump:
        run_one_case(args.dump, device)
    else:
        default = "/home/z00603376/ops-transformer-dev/attention/fused_infer_attention_score/tests/tests/input/layer_30_1775227380534.pt"
        run_one_case(default, device)

if __name__ == "__main__":
    main()
