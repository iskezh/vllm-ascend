#!/usr/bin/env python3
"""mssanitizer memcheck 驱动：prefill_l30 @ -3 真跳过（场景3）单次调用。"""
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
from run_dump_repro import _to_tensor, load_dump, run_custom_op  # noqa: E402

p = "/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps_v2/fia_layer30_rank0_chunked_prefill_occ1.pt"
inputs, params, _, _, _ = load_dump(p)
NH, NKV = params["num_heads"], params["num_kv_heads"]
SC, BS = params["scale"], params["block_size"]
sm = int(params.get("sparse_mode", 3))
q = inputs["query"].to("npu:0")
k = inputs["key"].to("npu:0")
v = inputs["value"].to("npu:0")
am = inputs["atten_mask"].to("npu:0")
aq = _to_tensor(inputs["actual_seq_lengths_q"], "npu:0")
akv = _to_tensor(inputs["actual_seq_lengths_kv"], "npu:0")
bt = inputs["block_table"].to("npu:0")

out, _, _ = run_custom_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, -3.0,
                          lse_flag=False, stats_flag=False)
torch.npu.synchronize()
print("memcheck driver done, nan=", int(torch.isnan(out).sum()), flush=True)
