#!/usr/bin/env python3
"""msprof 驱动：prefill_l30 dump 上循环跑 custom FIA kernel。

环境变量: LAM=sparse_lambda  STATS=0/1  ITERS=循环次数
"""
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

LAM = float(os.environ.get("LAM", "-99"))
STATS = os.environ.get("STATS", "0") == "1"
ITERS = int(os.environ.get("ITERS", "30"))
PATH = "/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps_v2/fia_layer30_rank0_chunked_prefill_occ1.pt"

inputs, params, _, _, meta = load_dump(PATH)
NH, NKV = params["num_heads"], params["num_kv_heads"]
SC, BS = params["scale"], params["block_size"]
sm = int(params.get("sparse_mode", 3))
dev = "npu:0"

q = inputs["query"].to(dev)
k = inputs["key"].to(dev)
v = inputs["value"].to(dev)
am = inputs["atten_mask"]
am = am.to(dev) if isinstance(am, torch.Tensor) and am.numel() > 0 else None
aq = _to_tensor(inputs["actual_seq_lengths_q"], dev)
akv = _to_tensor(inputs["actual_seq_lengths_kv"], dev)
bt = inputs["block_table"]
bt = bt.to(dev) if isinstance(bt, torch.Tensor) else None

print(f"driver: layer={meta['layer']} LAM={LAM} STATS={STATS} iters={ITERS}", flush=True)
for _ in range(3):  # warmup
    run_custom_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, LAM,
                  lse_flag=False, stats_flag=STATS)
torch.npu.synchronize()

for i in range(ITERS):
    run_custom_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, LAM,
                  lse_flag=False, stats_flag=STATS)
    torch.npu.synchronize()
print("driver done", flush=True)
