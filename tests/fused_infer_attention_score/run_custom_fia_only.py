#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小化复现脚本: 只调用 custom 算子 torch.ops._C_ascend.npu_fused_infer_attention_score。

固定加载 fia_layer20_rank0_decode_bigdiff_occ0.pt dump,
用于 mssanitizer (memcheck/racecheck) 单独分析该算子, 无命令行入参。
"""
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

DUMP_PATH = "/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps_v2/fia_layer20_rank0_decode_bigdiff_occ0.pt"
SWA_INT_MAX = 2147483647


def main():
    device = torch.device("npu:0")
    torch.npu.set_device(device)

    data = torch.load(DUMP_PATH, map_location="cpu", weights_only=False)
    inp, params = data["inputs"], data["params"]
    print(f"dump: {os.path.basename(DUMP_PATH)} "
          f"stage={data.get('stage', '?')} lambda={params['sparse_lambda']}",
          flush=True)

    q = inp["query"].to(device)
    k = inp["key"].to(device)
    v = inp["value"].to(device)
    am = inp["atten_mask"]
    am = am.to(device) if isinstance(am, torch.Tensor) and am.numel() > 0 else None
    bt = inp["block_table"]
    bt = bt.to(device) if isinstance(bt, torch.Tensor) and bt.numel() > 0 else None
    aq = inp["actual_seq_lengths_q"].to(device)
    akv = inp["actual_seq_lengths_kv"].to(device)

    with torch.npu.device(device):
        out, lse = torch.ops._C_ascend.npu_fused_infer_attention_score(
            q, k, v, None, am, aq, akv, bt,
            params["num_heads"], params["scale"],
            params.get("pre_tokens", SWA_INT_MAX),
            params.get("next_tokens", SWA_INT_MAX),
            "TND", params["num_kv_heads"], params["sparse_mode"],
            params.get("inner_precise", 0), params["block_size"], 0,
            params["sparse_lambda"], params.get("enable_lse_flag", False))
        torch.npu.synchronize()

    print(f"done. out={tuple(out.shape)} sum={out.float().sum().item():.6e}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
