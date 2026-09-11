#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""小 batch decode FD 边界验证: 从线上 dump (batch=11) 切出前 n 条请求重放。

背景: 线上 A/B (173948 run) 显示 decode 发散最严重集中在 tokens=3~6 的小 batch
(worst=(rel=1.17e4, tokens=3)), 怀疑 FD split/CombineScale 在小 batch 边界有问题。
本脚本把线上真实 dump (含真实 KV cache pool) 按请求数切分, 比较 custom vs baseline。

切分方式 (TND layout):
  q   : 前 sum(aq[:n]) 行
  bt  : block_table[:n]
  aq  : actual_seq_lengths_q[:n]
  akv : actual_seq_lengths_kv[:n]
  kv  : 完整 cache pool 不动 (block_table 索引)

用法:
    python test_subbatch_decode.py [--dump PATH] [--device npu:0]
"""

import argparse
import os

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


def run_custom(inp, params, device):
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
    q = inp["query"].to(device)
    k = inp["key"].to(device)
    v = inp["value"].to(device)
    am = inp["atten_mask"]
    am = am.to(device) if isinstance(am, torch.Tensor) and am.numel() > 0 else None
    bt = inp["block_table"]
    bt = bt.to(device) if isinstance(bt, torch.Tensor) and bt.numel() > 0 else None
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


def slice_batch(inp, n):
    """取前 n 条请求构成子 batch。aq 为累积长度, decode 每请求 q_len=1。"""
    aq = inp["actual_seq_lengths_q"]
    q_rows = int(aq[n - 1].item())
    return {
        "query": inp["query"][:q_rows],
        "key": inp["key"],
        "value": inp["value"],
        "atten_mask": inp["atten_mask"],
        "block_table": inp["block_table"][:n],
        "actual_seq_lengths_q": aq[:n],
        "actual_seq_lengths_kv": inp["actual_seq_lengths_kv"][:n],
    }


def diff_brief(a, b):
    ac, bc = a.detach().cpu(), b.detach().cpu()
    d = (ac.float() - bc.float()).abs()
    if ac.dtype == torch.bfloat16:
        n_bit = int((ac.view(torch.uint16) != bc.view(torch.uint16)).sum().item())
        # ULP 距离分布
        ai = ac.view(torch.uint16).to(torch.int32)
        bi = bc.view(torch.uint16).to(torch.int32)
        # bf16 单调映射: 负数翻转到单调区间
        am_ = torch.where(ai >= 0x8000, -ai, ai)  # 粗略, 仅看 >=5 ULP 计数
        bm_ = torch.where(bi >= 0x8000, -bi, bi)
        ulp = (am_ - bm_).abs()
        n_ulp5 = int((ulp >= 5).sum().item())
    else:
        n_bit = int((d > 0).sum().item())
        n_ulp5 = -1
    return d.max().item(), d.mean().item(), n_bit, n_ulp5, d.numel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", default="/home/z00603376/blasst_model/kvcomp/blasst_res/"
                   "attention_dumps_v2/fia_layer20_rank0_decode_bigdiff_occ0.pt")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--sizes", default="1,2,3,4,5,6,8,11")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.npu.set_device(device)

    data = torch.load(args.dump, map_location="cpu", weights_only=False)
    inp, params = data["inputs"], data["params"]
    full_bs = inp["actual_seq_lengths_q"].numel()
    print(f"dump: {os.path.basename(args.dump)}")
    print(f"  full batch={full_bs} akv={inp['actual_seq_lengths_kv'].tolist()}")
    print(f"  lambda={params['sparse_lambda']} sparse_mode={params['sparse_mode']}",
          flush=True)

    sizes = [int(s) for s in args.sizes.split(",") if 0 < int(s) <= full_bs]

    with torch.npu.device(device):
        for n in sizes:
            sub = slice_batch(inp, n)
            out_c = run_custom(sub, params, device)
            torch.npu.synchronize()
            out_b = run_base(sub, params, device)
            torch.npu.synchronize()
            mx, mean, nbit, nulp5, numel = diff_brief(out_c, out_b)
            print(f"[batch={n:2d} tokens={out_c.shape[0]:3d}] "
                  f"custom vs base: max={mx:.3e} mean={mean:.3e} "
                  f"bitdiff={nbit}/{numel} ulp>=5:{nulp5}", flush=True)


if __name__ == "__main__":
    main()
