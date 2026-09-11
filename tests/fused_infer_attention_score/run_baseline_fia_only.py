#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小化复现脚本: 只调用 baseline 算子 torch_npu.npu_fused_infer_attention_score。

固定加载 fia_layer20_rank0_decode_bigdiff_occ0.pt dump,
调用形式与 serving baseline 分支
(attention_v1.py forward_fused_infer_attention) 完全一致:
host list seq lens, 无 antiquant_mode, sparse_mode 取 dump 值。
用于 mssanitizer (memcheck/racecheck/initcheck) 单独分析该算子, 无命令行入参。
"""
import os
import sys

import torch
import torch_npu  # noqa: F401

DUMP_PATH = "/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps_v2/fia_layer20_rank0_decode_bigdiff_occ0.pt"


def main():
    device = torch.device("npu:0")
    torch.npu.set_device(device)

    data = torch.load(DUMP_PATH, map_location="cpu", weights_only=False)
    inp, params = data["inputs"], data["params"]
    print(f"dump: {os.path.basename(DUMP_PATH)} "
          f"stage={data.get('stage', '?')} sparse_mode={params['sparse_mode']}",
          flush=True)

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

    with torch.npu.device(device):
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query=q, key=k, value=v, atten_mask=am, block_table=bt,
            input_layout="TND", block_size=params["block_size"],
            actual_seq_lengths=aq, actual_seq_lengths_kv=akv,
            num_key_value_heads=params["num_kv_heads"],
            num_heads=params["num_heads"], scale=params["scale"],
            sparse_mode=params["sparse_mode"])
        torch.npu.synchronize()

    print(f"done. out={tuple(out.shape)} sum={out.float().sum().item():.6e}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
