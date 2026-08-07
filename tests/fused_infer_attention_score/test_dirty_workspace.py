#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue B 验证: 脏 workspace 是否会让 custom FIA 输出偏离基线。

假设: custom 算子存在 read-before-write 的 GM 读取
  H2: FD CombineScale 读取上次调用残留的 split 节点
  H3: PV GEMM 前读 gSp 稀疏标志 (StoreSpFlagResIntoHmb 在 softmaxReady 之后才发 MTE3 写,
      cube 端 DataCacheCleanAndInvalid 后读到的是旧值)
离线干净环境二者都读到 0/无害值 → bit-identical; 线上 workspace 池被反复搅动 → 发散。

方法: 用 0x01 / 0xFF 填充一大块 NPU 内存后释放, 让 caching allocator 把脏块
回收给 custom 算子的 workspace, 再跑 custom, 与基线对比。

用法:
    python test_dirty_workspace.py [--dump PATH] [--device npu:0]
"""

import argparse
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


def run_custom(inp, params, device):
    q = inp["query"].to(device)
    k = inp["key"].to(device)
    v = inp["value"].to(device)
    am = inp["atten_mask"]
    am = am.to(device) if isinstance(am, torch.Tensor) and am.numel() > 0 else None
    bt = inp["block_table"]
    bt = bt.to(device) if isinstance(bt, torch.Tensor) and bt.numel() > 0 else None
    aq = inp["actual_seq_lengths_q"].to(device)
    akv = inp["actual_seq_lengths_kv"].to(device)
    out, lse = torch.ops._C_ascend.npu_fused_infer_attention_score(
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


def diff_brief(a, b):
    ac, bc = a.detach().cpu(), b.detach().cpu()
    d = (ac.float() - bc.float()).abs()
    if ac.dtype == torch.bfloat16:
        n_bit = int((ac.view(torch.uint16) != bc.view(torch.uint16)).sum().item())
    else:
        n_bit = int((d > 0).sum().item())
    return d.max().item(), d.mean().item(), n_bit, d.numel()


def pollute(device, fill_byte, sizes_mb=(16, 24, 32, 48, 64, 96, 128, 256)):
    """分配一批填满 fill_byte 的大块再释放, 让 caching allocator 持有脏块。"""
    holders = []
    for mb in sizes_mb:
        t = torch.empty(mb * 1024 * 1024, dtype=torch.uint8, device=device)
        t.fill_(fill_byte)
        holders.append(t)
    torch.npu.synchronize()
    del holders  # 释放回 allocator, 内容保持脏
    torch.npu.synchronize()


def pollute_physical(device, fill_byte, total_mb=1024):
    """把 fill_byte 写满一大块物理 HBM 后 empty_cache 归还驱动,
    让 NPUWorkspaceAllocator 随后 AclrtMalloc 到这片脏物理页。"""
    holders = []
    for _ in range(total_mb // 64):
        t = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
        t.fill_(fill_byte)
        holders.append(t)
    torch.npu.synchronize()
    del holders
    torch.npu.empty_cache()  # aclrtFree, 物理页内容保持脏
    torch.npu.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", default="/home/z00603376/blasst_model/kvcomp/blasst_res/"
                   "attention_dumps_v2/fia_layer20_rank0_decode_bigdiff_occ0.pt")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.npu.set_device(device)

    data = torch.load(args.dump, map_location="cpu", weights_only=False)
    inp, params = data["inputs"], data["params"]
    print(f"dump: {os.path.basename(args.dump)}")
    print(f"  q={tuple(inp['query'].shape)} akv_max={inp['actual_seq_lengths_kv'].max().item()} "
          f"lambda={params['sparse_lambda']}", flush=True)

    with torch.npu.device(device):
        # 基线只跑一次 (npu 基线无 workspace 残留问题)
        out_n = run_base(inp, params, device)
        torch.npu.synchronize()

        # 控制组: 干净环境 custom
        out_c0 = run_custom(inp, params, device)
        torch.npu.synchronize()
        mx, mean, nbit, numel = diff_brief(out_c0, out_n)
        print(f"[control   ] custom vs base: max={mx:.3e} mean={mean:.3e} bitdiff={nbit}/{numel}",
              flush=True)

        # 实验组1: 0x01 脏 workspace (直接命中 gSp==1 skip 判定)
        for r in range(args.rounds):
            pollute(device, 0x01)
            out_c1 = run_custom(inp, params, device)
            torch.npu.synchronize()
            mx, mean, nbit, numel = diff_brief(out_c1, out_n)
            print(f"[dirty 0x01 #{r}] custom vs base: max={mx:.3e} mean={mean:.3e} "
                  f"bitdiff={nbit}/{numel}", flush=True)

        # 实验组2: 0xFF 脏 workspace (float 区为 NaN/garbage, gSp=255 不触发 skip)
        for r in range(args.rounds):
            pollute(device, 0xFF)
            out_c2 = run_custom(inp, params, device)
            torch.npu.synchronize()
            mx, mean, nbit, numel = diff_brief(out_c2, out_n)
            print(f"[dirty 0xFF #{r}] custom vs base: max={mx:.3e} mean={mean:.3e} "
                  f"bitdiff={nbit}/{numel}", flush=True)

        # 对照: custom 在两种污染下是否一致 (确认污染是否真被 custom 读到)
        pollute(device, 0x01)
        out_a = run_custom(inp, params, device)
        torch.npu.synchronize()
        pollute(device, 0xFF)
        out_b = run_custom(inp, params, device)
        torch.npu.synchronize()
        mx, mean, nbit, numel = diff_brief(out_a, out_b)
        print(f"[0x01 vs 0xFF] custom self-diff: max={mx:.3e} mean={mean:.3e} "
              f"bitdiff={nbit}/{numel}", flush=True)

        # 实验组3: 物理页级污染 —— empty_cache 归还驱动后, workspace AclrtMalloc 到脏页
        # 0x01: gSp 读到 1/1 → 触发 PV skip; split 节点读到 denormal
        for r in range(args.rounds):
            pollute_physical(device, 0x01)
            out_c3 = run_custom(inp, params, device)
            torch.npu.synchronize()
            mx, mean, nbit, numel = diff_brief(out_c3, out_n)
            print(f"[phys 0x01 #{r}] custom vs base: max={mx:.3e} mean={mean:.3e} "
                  f"bitdiff={nbit}/{numel}", flush=True)

        # 实验组4: 物理页级 0xFF (float 区 NaN)
        for r in range(args.rounds):
            pollute_physical(device, 0xFF)
            out_c4 = run_custom(inp, params, device)
            torch.npu.synchronize()
            mx, mean, nbit, numel = diff_brief(out_c4, out_n)
            print(f"[phys 0xFF #{r}] custom vs base: max={mx:.3e} mean={mean:.3e} "
                  f"bitdiff={nbit}/{numel}", flush=True)

        # 实验组5: λ=+1e30 全稀疏调用(把 gSp 写成 1)后立即跑 dense 调用,
        # 复用同一 workspace 池块, 检验 gSp 竞态是否被 cube 读到旧值
        params_sparse = dict(params)
        params_sparse["sparse_lambda"] = 1e30
        for r in range(args.rounds):
            _ = run_custom(inp, params_sparse, device)  # 全稀疏, gSp<-1
            torch.npu.synchronize()
            out_c5 = run_custom(inp, params, device)    # dense
            torch.npu.synchronize()
            mx, mean, nbit, numel = diff_brief(out_c5, out_n)
            print(f"[sparse->dense #{r}] custom vs base: max={mx:.3e} mean={mean:.3e} "
                  f"bitdiff={nbit}/{numel}", flush=True)


if __name__ == "__main__":
    main()
