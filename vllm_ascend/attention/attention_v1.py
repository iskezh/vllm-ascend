#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import logging
import os
from dataclasses import dataclass
from enum import Enum

import torch
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (  # type: ignore
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.registry import (  # type: ignore
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, CrossAttentionSpec

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.context_parallel.common_cp import AscendMetadataForDecode, AscendMetadataForPrefill
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    enable_cp,
    split_decodes_and_prefills,
    using_paged_attention,
)
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_graph_params,
    update_draft_graph_params_workspaces,
    update_graph_params_workspaces,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.flashcomm2_oshard_manager import flashcomm2_oshard_manager
from vllm_ascend.utils import weak_ref_tensors

# default max value of sliding window size
SWA_INT_MAX = 2147483647
_GLSE_DEBUG_PRINT = os.environ.get("VLLM_ASCEND_GLSE_DEBUG", "0") == "1"
# antiquant_mode = int(threshold * 10 - 100), default threshold = -3 => -130
_GLSE_ANTIQUANT_THRESHOLD = float(os.environ.get("VLLM_ASCEND_GLSE_ANTIQUANT_THRESHOLD", "-3"))
_GLSE_ANTIQUANT_MODE = int(_GLSE_ANTIQUANT_THRESHOLD * 10 - 100)
# 独立于 GLSE_DEBUG，仅传入 antiquant_mode，不打印不 dump
_ANTIQUANT_ONLY = os.environ.get("VLLM_ASCEND_ANTIQUANT_ONLY", "0") == "1"
# 是否使用迁移后的自定义 fused_infer_attention_score 算子分支
_USE_CUSTOM_FIA = os.environ.get("VLLM_ASCEND_USE_CUSTOM_FIA", "0") == "1"
_GLSE_DUMP_DECODE = os.environ.get("VLLM_ASCEND_GLSE_DUMP_DECODE", "0") == "1"
_GLSE_DUMP_DIR = os.environ.get(
    "VLLM_ASCEND_GLSE_DUMP_DIR",
    "/home/z00603376/blasst_model/kvcomp/glse_dump")
_GLSE_SAMPLE_INTERVAL = int(os.environ.get("VLLM_ASCEND_GLSE_SAMPLE_INTERVAL", "100"))
# 指定需要 dump 的 layer，逗号分隔，如 "20,30,50"
_GLSE_DUMP_TARGET_LAYERS = set()
_raw = os.environ.get("VLLM_ASCEND_GLSE_DUMP_LAYERS", "")
if _raw:
    _GLSE_DUMP_TARGET_LAYERS = {int(x.strip()) for x in _raw.split(",") if x.strip().isdigit()}
_glse_call_counter = 0
_glse_enable_lse_flag = False  # 每 N 次才开启 softmax_lse_flag
_glse_dumped_target_layers = set()  # 已 dump 的 target layer

# 控制 FIA 分支日志每个 layer 只打印一次
_fia_branch_logged_layers: set[str] = set()
# 控制 chunked_prefill dense 算子的单算子复现 dump（仅一次）
_dumped_chunked_prefill_dense = False

# ---- FIA 单算子复现 dump（debug，配置写死，需要时直接改这里） ----
_FIA_DUMP_ENABLE = True
_FIA_DUMP_DIR = "/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps_v2"
_FIA_DUMP_LAYERS = {0, 30, 63}   # 首/中/尾层
_FIA_DUMP_SKIP = 1               # 每个 (layer, stage) 组合跳过前 N 次（避开 warmup/profile）
_FIA_DUMP_COUNT = 1              # 跳过后再 dump N 次
_fia_dump_counter: dict = {}     # (layer_id, stage) -> 已出现次数
# NaN 陷阱：attn_output 首次出现 NaN 的层立即 dump（不受 _FIA_DUMP_LAYERS 限制，
# 每层只 dump 第一次），用于定位 NaN 起源层
_FIA_DUMP_ON_NAN = True
_fia_nan_dumped_layers: set = set()
# A/B 双跑对比：custom 先跑到临时 buffer，baseline 跑正式 output 并继续前传。
# custom 即使 NaN 也不污染残差流；每次调用对比两边 NaN，不一致即打印。
# False = custom 直接写正式 output 前传（纯 chunk prefill 已验证 100% bit 一致，真实投产形态）
_FIA_AB_COMPARE = True
# 全场景 A/B 采样模式：True 时 _can_use_custom_fia 放行所有 state（decode-only/混合/
# 纯 chunk 都进 custom 分支做 A/B 双跑），配合 _FIA_AB_STAGE_DUMP 抓三组对比数据。
# 仅定位用，投产必须 False。
_FIA_AB_ALL_STAGES = True
# 分场景采样 dump：decode-only / 混合 batch / 纯 chunk 各抓第一组（输入+双输出），
# 用于三场景单算子复现。每场景每进程限 1 次（首个遇到的层）。
_FIA_AB_STAGE_DUMP = True
_fia_stage_dumped: set = set()
_fia_ab_nan_events: list = []   # [(layer_name, stage, nt, nan_c, nan_b, maxdiff)]
# 数值对比聚合：stage -> dict；ulp 直方图桶 = [<1, 1, 2, 3, 4, >=5]
_fia_ab_stats: dict = {}
_FIA_AB_STAT_EVERY = 512      # 每 stage 每 N 次调用打印一次聚合汇总
_FIA_AB_REL_ALERT = 1e-2      # 非 NaN 调用 max_rel 超阈值即时告警
# 大差异陷阱：decode 调用 AB max_abs 超阈值即 dump 单算子复现数据（全局限 N 次/进程，
# 正常 ulp 噪声 max_abs<=0.06，线上坏 tile 为 38~85，阈值取 0.5）
_FIA_AB_BIGDIFF_DUMP = True
_FIA_AB_BIGDIFF_THRESH = 0.5
_FIA_AB_BIGDIFF_MAX = 2
_fia_bigdiff_dump_count = 0
# 保真 dump 模式：True 时 stage/bigdiff dump 整池 KV + 原始 block_table 原样落盘
# （不做 block 压缩 remap —— 压缩会丢掉陈旧列/原始池状态，恰好抹掉越读现场）。
# 每份约 2*num_blocks*block_size*num_kv_heads*head_size*2B（当前配置 ~728MB），
# 仅 stage dump(3 场景) 和 bigdiff dump(限 N 次) 走 raw；NaN-trap 逐层 dump 仍走
# 压缩模式，避免 64 层 × 8 rank 整池爆盘。
_FIA_DUMP_RAW = True
# 混合 batch grab 模式：True 时 _can_use_custom_fia 放行混合 batch 进 A/B 分支，
# 在跑任何 custom 算子之前先 raw dump 完整输入（崩溃也保留现场）。
_FIA_AB_MIXED_GRAB = True
# 混合 batch 是否线上跑 custom：custom regular kernel 在混合 batch 上曾致 worker
# 设备侧崩溃（20260803-194454 run），定位期一律 False —— 只抓输入 + baseline 输出，
# custom 行为靠离线单算子复现。
_FIA_AB_MIXED_RUN_CUSTOM = False
# 真实前传分场景路由（投产形态验证）：逗号分隔子集 {decode, chunk, mixed}，
# 仅列出的场景走 custom 真实输出，其余全部 baseline。设置后自动关闭 A/B 双跑
# 与各类 dump（真实路径，custom 写正式 output 继续前传）。
# 例：VLLM_ASCEND_FIA_CUSTOM_STAGES=decode  =>  chunk/mixed 走基线、decode 走 custom。
# 默认 "decode,chunk"：decode 与纯 chunk prefill 均走 custom（混合 batch 仍 baseline）。
_FIA_CUSTOM_STAGES = {
    s.strip() for s in os.environ.get("VLLM_ASCEND_FIA_CUSTOM_STAGES", "decode,chunk").split(",")
    if s.strip()
}
# A/B 角色反转：与 _FIA_CUSTOM_STAGES 配合，custom 写正式 output 前传，
# baseline 跑临时 buffer 仅做对比统计（DIFF/STAT 日志照常，dump 仍关闭）。
# 默认开启：STAGES 内场景 output 以 custom 为准，baseline 仅对比。
_FIA_AB_LOG = os.environ.get("VLLM_ASCEND_FIA_AB_LOG", "1") == "1"
if _FIA_CUSTOM_STAGES:
    _FIA_AB_ALL_STAGES = False
    _FIA_AB_STAGE_DUMP = False
    # BIGDIFF 陷阱保留开启：STAGES 模式下仍抓大差异 batch 的完整输入做离线复现。
    # 注意 _FIA_DUMP_ENABLE 必须保持 True —— _fia_repro_dump 入口强制检查它，
    # 否则 BIGDIFF 的 force dump 会被静默跳过（20260806 101328 run 曾因此只落了
    # outputs 小文件、丢了输入复现包）。NaN/STAGE dump 仍由各自开关保持关闭。
    _FIA_DUMP_ENABLE = True
    _FIA_DUMP_ON_NAN = False
    _FIA_AB_COMPARE = _FIA_AB_LOG
    _FIA_AB_CUSTOM_REAL = _FIA_AB_LOG
else:
    _FIA_AB_CUSTOM_REAL = False
logger = logging.getLogger(__name__)

# print(f"[FIA_BRANCH] attention_v1 module loaded: _USE_CUSTOM_FIA={_USE_CUSTOM_FIA}", flush=True)


def _get_stage_name(attn_state):
    """Map AscendAttentionState to human-readable stage name for logging."""
    stage_map = {
        0: "prefill",           # PrefillNoCache
        1: "prefill_cache",     # PrefillCacheHit
        2: "decode",            # DecodeOnly
        3: "chunked_prefill",   # ChunkedPrefill
        4: "spec_decode",       # SpecDecoding
    }
    raw = attn_state.value if hasattr(attn_state, "value") else int(attn_state)
    return stage_map.get(raw, "unknown")


_glse_dumped_layers = set()

# 用于 _forward_c8_chunked_prefill 中只打印一次分支走向
_chunked_prefill_branch_logged = set()


def _log_chunked_prefill_branch(layer_name, all_new_prefill, float_key, float_value, attn_metadata):
    """打印 C8 ChunkedPrefill 的 prefill 分支选择，帮助确认是否走了新分支。"""
    key = (layer_name, all_new_prefill, float_key is not None, float_value is not None)
    if key in _chunked_prefill_branch_logged:
        return
    _chunked_prefill_branch_logged.add(key)

    branch = "NEW(float_kv_tnd)" if (all_new_prefill and float_key is not None and float_value is not None) else "LEGACY(gather_dequant)"
    # print(
    #     f"[C8_CHUNKED_PREFILL][{branch}] layer={layer_name} "
    #     f"attn_state={attn_metadata.attn_state.name} "
    #     f"num_decodes={attn_metadata.num_decodes} "
    #     f"num_prefills={attn_metadata.num_prefills} "
    #     f"num_decode_tokens={attn_metadata.num_decode_tokens} "
    #     f"all_new_prefill={all_new_prefill} "
    #     f"float_key_present={float_key is not None} "
    #     f"float_value_present={float_value is not None} "
    #     f"antiquant_threshold={_GLSE_ANTIQUANT_THRESHOLD} "
    #     f"antiquant_mode={_GLSE_ANTIQUANT_MODE} "
    #     f"use_custom_fia={_USE_CUSTOM_FIA}"
    # )


def _glse_need_dump_layer(layer_name):
    """检查当前 layer 是否是 dump target 且尚未 dump。"""
    import re
    if not _GLSE_DUMP_DECODE or not _GLSE_DUMP_TARGET_LAYERS:
        return False
    m = re.search(r"layers\.(\d+)", layer_name)
    if m:
        lid = int(m.group(1))
        return lid in _GLSE_DUMP_TARGET_LAYERS and lid not in _glse_dumped_target_layers
    return False


def _glse_should_sample(layer_name=""):
    """每 _GLSE_SAMPLE_INTERVAL 次调用返回 True，或当前 layer 是 dump target 且未 dump 过。

    返回 True 时 softmax_lse_flag 开启，可获取 gLse 数据用于打印或 dump。
    """
    global _glse_call_counter, _glse_enable_lse_flag
    _glse_call_counter += 1
    periodic = (_glse_call_counter % _GLSE_SAMPLE_INTERVAL == 0)
    is_target = _glse_need_dump_layer(layer_name)
    _glse_enable_lse_flag = periodic or is_target
    return _glse_enable_lse_flag


def _glse_brief_print(layer_name, stage, sample=False):
    """简单打印：layer + stage + antiquant_mode。sample=True 时标明本次采样 gLse。"""
    tag = "[SAMPLE]" if sample else ""
    # print(f"[GLSE{tag}][{stage}] layer={layer_name} "
    #       f"antiquant_mode={_GLSE_ANTIQUANT_MODE} call={_glse_call_counter}")


def _dump_decode_tensors(softmax_lse, layer_name, stage, **tensors):
    """Dump decode attention tensors to .pt for offline single-op reproduction.

    Only dumps every 5th layer (layer_id % 5 == 0), and each layer only
    on first non-integer hit to avoid excessive disk usage.
    """
    import time
    import re

    try:
        match = re.search(r"layers\.(\d+)", layer_name)
        layer_id = int(match.group(1)) if match else -1
        # target layer dump: 不检查旧限制，直接 dump 一次
        # 只有 target layer 才 dump
        if layer_id not in _GLSE_DUMP_TARGET_LAYERS:
            return
        if layer_id in _glse_dumped_target_layers:
            return
        _glse_dumped_target_layers.add(layer_id)

        os.makedirs(_GLSE_DUMP_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)
        pid = os.getpid()
        filename = f"layer_{layer_id}_{timestamp}_pid{pid}.pt"
        dump_path = os.path.join(_GLSE_DUMP_DIR, filename)

        dump_data = {"layer_id": layer_id, "inputs": {}, "outputs": {}}
        for k, v in tensors.items():
            if k == "attn_output":
                dump_data["outputs"]["attn_output"] = v.cpu() if isinstance(v, torch.Tensor) else v
            elif isinstance(v, torch.Tensor):
                dump_data["inputs"][k] = v.cpu()
            elif isinstance(v, (int, float)):
                dump_data["inputs"][k] = torch.tensor(v)
            else:
                dump_data["inputs"][k] = v
        dump_data["outputs"]["softmax_lse"] = softmax_lse.cpu()

        torch.save(dump_data, dump_path)
        # print(f"[GLSE][{stage}] layer={layer_name} DUMPED tensors to {dump_path}")
    except Exception as e:
        # print(f"[GLSE][{stage}] layer={layer_name} DUMP FAILED: {e}")
        pass


def _dump_chunked_prefill_dense_inputs(layer_name, stage, sparseblock, totalblock, **tensors):
    """Dump 单算子复现用的 chunked_prefill dense 输入数据 (ops-transformer-dev 格式)。

    当 chunked_prefill 阶段 sparseblock==0（dense 模式）时，dump
    全部输入到 .pt 文件。仅 dump 一次，避免重复。

    输出格式兼容 ops-transformer-dev 的 test_accuracy_with_rowloop_sparse_v1.py:
        data["inputs"] = {query, key, value, block_table, block_size,
                          actual_seq_qlen, actual_seq_kvlen,
                          num_heads, num_kv_heads, scale}
        data["outputs"] = {attn_output}
        data["layer_id"] = int
    其中标量参数存为 0-dim tensor。
    """
    global _dumped_chunked_prefill_dense
    if _dumped_chunked_prefill_dense:
        return
    _dumped_chunked_prefill_dense = True

    import time, re

    try:
        os.makedirs(_GLSE_DUMP_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)
        pid = os.getpid()
        filename = f"layer_30_{timestamp}_pid{pid}.pt"
        dump_path = os.path.join(_GLSE_DUMP_DIR, filename)

        # Extract layer_id from layer_name
        m = re.search(r"layers\.(\d+)", layer_name)
        layer_id = int(m.group(1)) if m else -1

        # Build inputs dict matching ops-transformer-dev format
        dump_inputs = {}
        scalar_keys = {
            "num_heads": "num_heads", "num_kv_heads": "num_kv_heads",
            "scale": "scale", "block_size": "block_size",
        }
        tensor_keys = ["query", "key", "value"]
        for k in tensor_keys:
            v = tensors.get(k)
            dump_inputs[k] = v.detach().cpu() if isinstance(v, torch.Tensor) else v

        # block_table: if None or not passed, store empty tensor
        bt = tensors.get("block_table")
        if bt is None:
            bt = torch.empty(0, dtype=torch.float32)
        elif isinstance(bt, torch.Tensor):
            bt = bt.detach().cpu()
        dump_inputs["block_table"] = bt

        # Scalar params as 0-dim tensors (matching ops-transformer-dev style)
        for iname, k in scalar_keys.items():
            v = tensors.get(k)
            if isinstance(v, torch.Tensor):
                dump_inputs[iname] = v.detach().cpu()
            else:
                dump_inputs[iname] = torch.tensor(v)

        # Sequence lengths as 1-dim tensors
        for iname, k in [("actual_seq_qlen", "actual_seq_lengths_q"),
                         ("actual_seq_kvlen", "actual_seq_lengths_kv")]:
            v = tensors.get(k)
            if isinstance(v, torch.Tensor):
                dump_inputs[iname] = v.detach().cpu()
            elif isinstance(v, list):
                dump_inputs[iname] = torch.tensor(v, dtype=torch.int64)
            else:
                dump_inputs[iname] = torch.tensor([v], dtype=torch.int64)

        attn_out = tensors.get("attn_output")
        if attn_out is not None and isinstance(attn_out, torch.Tensor):
            attn_out = attn_out.detach().cpu()
        dump_outputs = {"attn_output": attn_out}

        dump_data = {
            "layer_id": layer_id,
            "inputs": dump_inputs,
            "outputs": dump_outputs,
        }
        torch.save(dump_data, dump_path)
        # print(f"[GLSE][{stage}] layer={layer_name} "
        #       f"DUMPED chunked_prefill_dense inputs (sparse={sparseblock}/{totalblock}) to {dump_path}")
    except Exception as e:
        # print(f"[GLSE][{stage}] layer={layer_name} DUMP chunked_prefill_dense FAILED: {e}")
        # import traceback; traceback.print_exc()
        pass


def _fia_repro_dump(layer_name, stage, num_tokens, attn_output, query, key,
                    value, atten_mask, actual_seq_lengths_q,
                    actual_seq_lengths_kv, block_table, num_heads,
                    num_kv_heads, scale, sparse_mode, pre_tokens, next_tokens,
                    block_size, sparse_lambda, enable_lse_flag,
                    force=False, tag="", raw=False):
    """Dump custom-FIA 算子输入，供 run_dump_repro.py(格式A) 离线复现对比。

    - 每个 (_FIA_DUMP_LAYERS 中的 layer, stage) 组合：跳过前 _FIA_DUMP_SKIP 次，
      再 dump _FIA_DUMP_COUNT 次。force=True 时跳过过滤与计数（NaN 陷阱用）。
    - raw=False（默认）：paged 路径对 KV cache 做 block 压缩重映射，只保留本 batch
      实际引用的 block —— 但这会丢掉陈旧列与原始池状态，恰好抹掉越读现场。
    - raw=True（保真模式，stage/bigdiff dump 用）：整池 KV + 原始 block_table
      原样落盘，不做任何 remap，完整保留陈旧列/垃圾块等越读触发条件。
      attn_output 允许为 None（pre-op dump，输出见 fia_stage_outputs_*.pt）。
    """
    if not _FIA_DUMP_ENABLE:
        return
    import re
    m = re.search(r"layers\.(\d+)", layer_name)
    if not m:
        return
    layer_id = int(m.group(1))
    combo = (layer_id, stage)
    seen = _fia_dump_counter.get(combo, 0)
    if not force:
        if layer_id not in _FIA_DUMP_LAYERS:
            return
        _fia_dump_counter[combo] = seen + 1
        if seen < _FIA_DUMP_SKIP or seen >= _FIA_DUMP_SKIP + _FIA_DUMP_COUNT:
            return

    try:
        os.makedirs(_FIA_DUMP_DIR, exist_ok=True)
        try:
            from vllm.distributed.parallel_state import get_tp_group
            tp_rank = get_tp_group().rank_in_group
        except Exception:
            tp_rank = os.getpid()

        def _cpu(t):
            return t.detach().cpu() if isinstance(t, torch.Tensor) else t

        akv_list = (actual_seq_lengths_kv.detach().cpu().tolist()
                    if isinstance(actual_seq_lengths_kv, torch.Tensor)
                    else [int(x) for x in actual_seq_lengths_kv])

        # --- KV cache block 处理 ---
        # raw=True：整池 + 原始 block_table 原样落盘（保真，供越读问题复现）。
        # raw=False：block 压缩重映射。注意：收集 bt 整行（含超过 nb 的陈旧列）
        # 引用的所有 block。kernel 循环存在 kvEnd + preKVNum 越读，陈旧列指向的
        # block 也可能被读到 —— 压缩模式仅用于 NaN-trap 等只关心数值的场合。
        bt_meta = None
        if block_table is not None:
            bt_cpu = block_table.detach().cpu()
            bs = int(block_size)
            batch = min(len(akv_list), bt_cpu.shape[0])
            nb_list = [(int(akv_list[b]) + bs - 1) // bs for b in range(batch)]
            bt_meta = {
                "orig_shape": list(block_table.shape),
                "orig_stride": list(block_table.stride()),
                "orig_contiguous": bool(block_table.is_contiguous()),
                "raw": bool(raw),
            }
            if raw:
                k_d = key.detach().cpu()
                v_d = value.detach().cpu()
                bt_d = bt_cpu
            else:
                full_rows = [bt_cpu[b, :].long() for b in range(batch)]
                nblk = key.shape[0]
                valid_rows = [r[(r >= 0) & (r < nblk)] for r in full_rows]
                used_ids = torch.unique(torch.cat(valid_rows)) if valid_rows else \
                    torch.empty(0, dtype=torch.long)
                k_full = key.detach().cpu()
                v_full = value.detach().cpu()
                k_d = k_full[used_ids].clone()
                v_d = v_full[used_ids].clone()
                remap = torch.full((k_full.shape[0], ), -1, dtype=torch.long)
                remap[used_ids] = torch.arange(used_ids.numel(), dtype=torch.long)
                width = bt_cpu.shape[1]
                new_bt = torch.full((batch, width), -1, dtype=torch.long)
                for b, r in enumerate(full_rows):
                    m = (r >= 0) & (r < nblk)
                    new_bt[b][m] = remap[r[m]]
                bt_d = new_bt.to(bt_cpu.dtype)
        else:
            nb_list = None
            k_d = _cpu(key)
            v_d = _cpu(value)
            bt_d = None

        dump_data = {
            "inputs": {
                "query": _cpu(query),
                "key": k_d,
                "value": v_d,
                "atten_mask": _cpu(atten_mask),
                "block_table": bt_d,
                "actual_seq_lengths_q":
                    actual_seq_lengths_q.detach().cpu()
                    if isinstance(actual_seq_lengths_q, torch.Tensor)
                    else torch.tensor(actual_seq_lengths_q, dtype=torch.int64),
                "actual_seq_lengths_kv": torch.tensor(akv_list,
                                                      dtype=torch.int64),
            },
            "params": {
                "num_heads": int(num_heads),
                "num_kv_heads": int(num_kv_heads),
                "scale": float(scale),
                "sparse_mode": int(sparse_mode),
                "inner_precise": 0,
                "pre_tokens": int(pre_tokens),
                "next_tokens": int(next_tokens),
                "block_size": int(block_size),
                "sparse_lambda": float(sparse_lambda),
                "enable_lse_flag": bool(enable_lse_flag),
            },
            "outputs": {
                "attn_output": _cpu(attn_output),
            },
            "stage": stage,
            "layer_name": layer_name,
            "layer_id": layer_id,
            "tp_rank": tp_rank,
            "num_tokens": int(num_tokens),
            "bt_meta": bt_meta,
            "nb_list": nb_list if block_table is not None else None,
            "q_stride": list(query.stride()),
            "q_contiguous": bool(query.is_contiguous()),
        }
        dump_data["query_has_nan"] = bool(
            torch.isnan(dump_data["inputs"]["query"]).any().item())
        dump_data["key_has_nan"] = bool(torch.isnan(k_d).any().item())
        fname = (f"fia_layer{layer_id:02d}_rank{tp_rank}_{stage}{tag}"
                 f"_occ{seen}.pt")
        torch.save(dump_data, os.path.join(_FIA_DUMP_DIR, fname))
        print(f"[FIA_DUMP] saved {fname} q={tuple(dump_data['inputs']['query'].shape)} "
              f"kv_blocks={None if bt_d is None else k_d.shape[0]}",
              flush=True)
    except Exception as e:
        print(f"[FIA_DUMP] dump FAILED layer={layer_name} stage={stage}: {e}",
              flush=True)


def _parse_glse_block_info(softmax_lse, layer_name, stage="", **dump_kwargs):
    """Parse gLse tensor to extract sparse block and total block counts.

    Follows the GLSE parsing pattern from the FIA precision test
    (test_accuracy_with_rowloop_sparse_v1.py).
    Prints: [GLSE][stage] layer=layer_name sparseblock=N / totalblock=M

    If any value is non-integer, prints "0 / 0".
    When _GLSE_DUMP_DECODE is set and stage=="decode" and non-int detected,
    dumps tensors via _dump_decode_tensors.

    Returns (sparse_int, block_int) on success, (0, 0) on parse failure.
    """
    try:
        data_1d = softmax_lse.flatten().cpu().float()
        # print(f"[GLSE][{stage}] layer={layer_name} glse_len={len(data_1d)} shape={softmax_lse.shape}")
        coreNum = min(24, len(data_1d) // 16)
        blockSparseNum = 0.0
        blockNum = 0.0
        core_details = []
        for core in range(coreNum):
            sparse_c = data_1d[core * 16].item()
            count_c = data_1d[core * 16 + 1].item()
            blockSparseNum += sparse_c
            blockNum += count_c
            core_details.append(f"c{core}:s={sparse_c:.2f}/c={count_c:.2f}")
        # Print per-core raw values for debugging
        # print(f"[GLSE][{stage}] layer={layer_name} per_core: {' | '.join(core_details)}")
        # Check if both are effectively integers
        sparse_int = int(round(blockSparseNum))
        block_int = int(round(blockNum))
        if (abs(blockSparseNum - sparse_int) > 1e-4
                or abs(blockNum - block_int) > 1e-4
                or block_int == 0):
            print(f"[GLSE][{stage}] layer={layer_name} sparseblock=0 / totalblock=0 (non_int sparseSum={blockSparseNum:.2f} blockSum={blockNum:.2f})")
            if _GLSE_DUMP_DECODE and stage == "decode" and dump_kwargs:
                _dump_decode_tensors(softmax_lse, layer_name, stage, **dump_kwargs)
            return 0, 0
        else:
            # print(f"[GLSE][{stage}] layer={layer_name} sparseblock={sparse_int} / totalblock={block_int}")
            # 对于 target layer，即使 gLse 正常也 dump
            if (_GLSE_DUMP_DECODE and stage == "decode" and dump_kwargs
                    and _glse_need_dump_layer(layer_name)):
                _dump_decode_tensors(softmax_lse, layer_name, stage, **dump_kwargs)
            return sparse_int, block_int
    except Exception as e:
        # print(f"[GLSE][{stage}] layer={layer_name} sparseblock=0 / totalblock=0 err={e}")
        return 0, 0


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "CUSTOM" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPImpl

            return AscendAttentionCPImpl
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPMetadataBuilder

            return AscendAttentionCPMetadataBuilder
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendMetadata:
    """
    Per-layer attention metadata for Ascend FlashAttention backend.

    Contains attention masks, token counts, sequence lengths and KV cache
    related properties for attention computation.
    """

    # **************************** Basic Properties ************************** #
    attn_mask: torch.Tensor | None = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens_pcp_padded: int = 0
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0
    num_decodes_flatten: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    # TODO(Angazenn): The following parameters are quite redundant and
    # contains similar information (such as seq_lens seq_lens_list). We
    # should simplified these parameters once attention schema in vLLM-Ascend
    # is unified.
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    seq_lens_list: list[int] = None  # type: ignore
    actual_seq_lengths_q: list[int] = None  # type: ignore

    query_start_loc: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: int | None = None

    # ********************** KV Cache Related Properties ********************* #
    # Block addresses per sequence (Seq id -> list of physical block).
    # (batch_size, max_blocks_per_seq)
    block_tables: torch.Tensor = None

    # The indices of the token slots that input tokens will be stored into.
    # E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the
    # three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0,
    # and 1st slot in block 1, respectively.
    # (num_tokens,)
    slot_mapping: torch.Tensor = None
    # pcp
    prefill: AscendMetadataForPrefill | None = None
    # dcp
    decode_meta: AscendMetadataForDecode | None = None

    causal: bool = True
    # runner_type in model_config.
    model_runner_type: str = ""
    # prefill reshape_and_cache event
    reshape_cache_event: torch.npu.Event = None

    # sliding window attention mask
    swa_mask: torch.Tensor | None = None


class AscendAttentionMetadataBuilder(AttentionMetadataBuilder[AscendMetadata]):
    """
    Builder for constructing AscendMetadata from CommonAttentionMetadata.

    Handles attention mask generation and metadata preparation for
    Ascend FlashAttention backend.
    """

    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.compilation_config = vllm_config.compilation_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len, AscendAttentionBackend.get_supported_kernel_block_sizes()[0]
        )

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        self.reorder_batch_threshold = self.decode_threshold

        scheduler_config = vllm_config.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendAttentionMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.ALWAYS

    def reorder_batch(self, input_batch, scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.decode_threshold
        )

        block_table = common_attn_metadata.block_table_tensor
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]

        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        # this slot_mapping override doesn't work since vllm will override it again. We should fix it vllm.
        # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
        if isinstance(self.kv_cache_spec, CrossAttentionSpec):
            seq_lens = common_attn_metadata.seq_lens
            slot_mapping = common_attn_metadata.slot_mapping.to(torch.int32)
        elif self.speculative_config and self.speculative_config.parallel_drafting:
            seq_lens = common_attn_metadata.seq_lens

        attn_state = common_attn_metadata.attn_state

        # Get attn_mask and swa_mask from singleton AttentionMaskBuilder
        attn_mask = self.attn_mask_builder.get_attention_mask(self.model_config)

        swa_mask = None
        is_swa = hasattr(self.model_config.hf_text_config, "sliding_window")
        if self.model_config is not None and is_swa:
            swa_mask = self.attn_mask_builder.get_swa_mask(
                self.model_config.dtype, self.model_config.hf_text_config.sliding_window
            )

        # TODO: Yet another unnecessary H2D while we already have a query_start_loc on device
        query_start_loc = query_start_loc_cpu.pin_memory().to(self.device, non_blocking=True)

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            seq_lens_list=seq_lens.tolist(),
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            swa_mask=swa_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            causal=common_attn_metadata.causal,
            model_runner_type=self.model_config.runner_type,
        )
        return attn_metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.ChunkedPrefill,
            AscendAttentionState.SpecDecoding,
        ):
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and ChunkedPrefill state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendAttentionBackendImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        sinks: torch.Tensor = None,
        **kwargs,
    ) -> None:
        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32, device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        self.is_kv_producer = (
            self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        )
        self.sinks = sinks

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
        draft_attn_metadatas=None,
    ):
        if using_paged_attention(num_tokens, vllm_config):
            # Paged Attention update logic
            if _EXTRA_CTX.is_draft_model:
                graph_params = get_draft_graph_params()
            else:
                graph_params = get_graph_params()
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    forward_context.attn_metadata,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value_cache,
                        num_kv_heads,
                        num_heads,
                        scale,
                        block_table,
                        seq_lens,
                        output,
                    ) = param
                    seq_lens = forward_context.attn_metadata[key].seq_lens

                    workspace = torch_npu._npu_paged_attention_get_workspace(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                    )
                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu._npu_paged_attention(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                        workspace=workspace,
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
        else:
            # FIA update logic
            if _EXTRA_CTX.is_draft_model:
                graph_params = get_draft_graph_params()
                attn_metadata = draft_attn_metadatas
                attn_keys = list(attn_metadata[0].keys())
            else:
                graph_params = get_graph_params()
                attn_metadata = forward_context.attn_metadata
                attn_keys = list(attn_metadata.keys())
            # For Qwen3-next, since the kv_cache_config has already categorized
            # linear_attn and self_attn, the attn_metadata is first arranged with
            # self_attn followed by linear_attn. Therefore, using zip directly
            # filters out the update operations for linear_attn.
            # TODO: We use a new variable `attn_keys` to ensure the loop count is
            # correct after get by `zip` because of the new structure of the attn_metadata
            # when running with the merged full eagle-graph. Should check it with Qwen3-next.
            num_layers = len(attn_keys)
            if num_layers == 0:
                return
            if _EXTRA_CTX.is_draft_model:
                attn_keys = attn_keys * (len(graph_params.attn_params[num_tokens]) // num_layers)
            attn_count = 0
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    attn_keys,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value,
                        block_tables,
                        attn_mask,
                        block_size,
                        seq_lens,
                        query_start_loc,
                        num_kv_heads,
                        num_heads,
                        scale,
                        attn_output,
                        softmax_lse,
                    ) = param

                    if _EXTRA_CTX.is_draft_model:
                        draft_step = attn_count // num_layers
                        seq_lens = attn_metadata[draft_step][key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[draft_step][key].actual_seq_lengths_q
                        block_tables = attn_metadata[draft_step][key].block_tables
                        attn_count = attn_count + 1
                    else:
                        seq_lens = attn_metadata[key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[key].actual_seq_lengths_q
                        block_tables = attn_metadata[key].block_tables

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu.npu_fused_infer_attention_score.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_lengths=actual_seq_lengths_q,
                        actual_seq_lengths_kv=seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale=scale,
                        sparse_mode=3,
                        workspace=graph_params.workspaces.get(num_tokens),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)

                    event.record(update_stream)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        if flashcomm2_oshard_manager.flashcomm2_oshard_enable():
            flashcomm2_oshard_manager.post_process_after_loading()

    def full_graph_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)

        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if _EXTRA_CTX.is_draft_model:
            graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        # Prepare tensors for attention output
        # TODO: Refactor this to step-level instead of layer-level

        # Get workspace from cache or calculate it if not present.
        workspace = graph_params.workspaces.get(num_tokens)
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=3,
                scale=self.scale,
            )
            if _EXTRA_CTX.is_draft_model:
                update_draft_graph_params_workspaces(num_tokens, workspace)
            else:
                update_graph_params_workspaces(num_tokens, workspace)

        # Handle graph capturing mode
        stream = torch_npu.npu.current_stream()

        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attn_params[num_tokens].append(
            (
                weak_ref_tensors(query),
                weak_ref_tensors(key),
                weak_ref_tensors(value),
                weak_ref_tensors(block_table),
                weak_ref_tensors(attn_metadata.attn_mask),
                block_size,
                actual_seq_lengths_kv,
                actual_seq_lengths_q,
                self.num_kv_heads,
                self.num_heads,
                self.scale,
                weak_ref_tensors(output),
                weak_ref_tensors(softmax_lse),
            )
        )

        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
            workspace=workspace,
            out=[output, softmax_lse],
        )

        output = output.view(num_tokens, self.num_heads, self.head_size)

        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        return output, num_tokens

    def full_graph_pa(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ):
        graph_params = get_graph_params()
        num_tokens = query.shape[0]
        if _EXTRA_CTX.capturing:
            # Get workspace from cache or calculate it if not present.
            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                workspace = torch_npu._npu_paged_attention_get_workspace(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output,
                )
                update_graph_params_workspaces(num_tokens, workspace)

            # Handle graph capturing mode
            stream = torch_npu.npu.current_stream()

            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)
            graph_params.attn_params[num_tokens].append(
                (
                    weak_ref_tensors(query),
                    weak_ref_tensors(self.key_cache),
                    weak_ref_tensors(self.value_cache),
                    self.num_kv_heads,
                    self.num_heads,
                    self.scale,
                    attn_metadata.block_tables,
                    attn_metadata.seq_lens,
                    weak_ref_tensors(output),
                )
            )

            torch.npu.graph_task_group_begin(stream)
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=attn_metadata.block_tables,
                context_lens=attn_metadata.seq_lens,
                out=output,
                workspace=workspace,
            )
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
            return output

    def _get_fia_params(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata):
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            block_size = 128
            block_table = None
            actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
            if self.attn_type == AttentionType.ENCODER_DECODER:
                actual_seq_lengths_kv = torch.cumsum(attn_metadata.seq_lens, dim=0).tolist()
        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            batch_size = attn_metadata.seq_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        # chunked prefill.
        else:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        return key, value, block_size, block_table, actual_seq_lengths_kv

    def _forward_fia_slidingwindow(self, query: torch.Tensor, attn_metadata: AscendMetadata, output: torch.Tensor):
        batch_size = attn_metadata.seq_lens.shape[0]
        block_size = 128
        query = query.view(batch_size, 1, self.num_heads * self.head_size)
        key = self.key_cache
        value = self.value_cache
        if self.key_cache is not None and self.value_cache is not None:
            block_size = self.key_cache.shape[1]
            key = self.key_cache.flatten(2, 3).contiguous()
            value = self.value_cache.flatten(2, 3).contiguous()

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BSH",
            block_size=block_size,
            pre_tokens=self.sliding_window,
            scale=self.scale,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths=[1] * len(attn_metadata.seq_lens),
            actual_seq_lengths_kv=attn_metadata.seq_lens,
        )

        attn_output = attn_output.view(batch_size, self.num_heads, self.head_size)
        output[:batch_size] = attn_output[:batch_size]
        return output

    def _can_use_custom_fia(
        self,
        attn_metadata: AscendMetadata,
    ) -> bool:
        """Check whether the current attention config can use the migrated custom FIA op.

        The custom op currently only supports TND layout, sparse_mode 0/3, and does
        not provide learnable_sink support or a graph-capture `.out` variant.
        """
        if not _USE_CUSTOM_FIA:
            return False
        if _EXTRA_CTX.capturing:
            return False
        if self.sinks is not None:
            return False
        if self.sliding_window is not None:
            return False
        # 真实前传分场景路由模式：仅 _FIA_CUSTOM_STAGES 列出的场景走 custom。
        # _FIA_AB_LOG=1 时 decode/纯 chunk 无条件进 A/B 分支：STAGES 内的场景
        # custom 写正式 output，其余场景 baseline 写正式 output、custom 仅对比。
        if _FIA_CUSTOM_STAGES:
            _st = attn_metadata.attn_state
            if _st == AscendAttentionState.DecodeOnly:
                return "decode" in _FIA_CUSTOM_STAGES or _FIA_AB_LOG
            if _st == AscendAttentionState.PrefillNoCache:
                # 首 chunk 无 cache 的 prefill：按 "prefill"（或兼容 "chunk"）路由
                return ("prefill" in _FIA_CUSTOM_STAGES
                        or "chunk" in _FIA_CUSTOM_STAGES)
            if _st == AscendAttentionState.ChunkedPrefill:
                aq = attn_metadata.actual_seq_lengths_q
                if aq:
                    q_lens = [aq[0]] + [aq[i] - aq[i - 1] for i in range(1, len(aq))]
                    if min(q_lens) == 1 and max(q_lens) > 1:
                        return "mixed" in _FIA_CUSTOM_STAGES
                return "chunk" in _FIA_CUSTOM_STAGES or _FIA_AB_LOG
            return False
        # 全场景 A/B 采样模式：除混合 batch 外所有 state 进 custom 分支（custom 写
        # 临时 buffer，正式前传仍是 baseline），抓 decode-only/纯 chunk 对比数据。
        # 混合 batch：_FIA_AB_MIXED_GRAB=True 时也进 A/B 分支，但仅在跑 custom 前
        # raw dump 输入（内部按 _FIA_AB_MIXED_RUN_CUSTOM 决定是否线上跑 custom，
        # 默认不跑 —— custom regular kernel 在混合 batch 上不仅产生 NaN/错值，
        # 还曾致 worker 进程设备侧崩溃（20260803-194454 run））。
        if _FIA_AB_ALL_STAGES:
            if attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
                aq = attn_metadata.actual_seq_lengths_q
                if aq:
                    q_lens = [aq[0]] + [aq[i] - aq[i - 1] for i in range(1, len(aq))]
                    if min(q_lens) == 1 and max(q_lens) > 1:
                        return _FIA_AB_MIXED_GRAB
            return True
        # 仅纯 chunk prefill 走 custom：decode-only 的 custom FD kernel 线上存在
        # 间歇性整行错值（离线全部 bit 一致，线上污染根因未定位），DecodeOnly 回退
        # baseline；混合 batch custom regular kernel 有状态依赖竞争（NaN/整行错值），
        # 同样回退 baseline。
        if attn_metadata.attn_state != AscendAttentionState.ChunkedPrefill:
            return False
        # actual_seq_lengths_q 为前缀和，逐请求 q_len = 相邻差分；
        # min q_len==1 && max q_len>1 即 decode 与 prefill chunk 混合 batch。
        aq = attn_metadata.actual_seq_lengths_q
        if aq:
            q_lens = [aq[0]] + [aq[i] - aq[i - 1] for i in range(1, len(aq))]
            if min(q_lens) == 1 and max(q_lens) > 1:
                return False
        return True

    def forward_custom_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer_name: str = "",
    ):
        """Forward path through the migrated custom FIA operator.

        This mirrors ``forward_fused_infer_attention`` but calls
        ``torch.ops._C_ascend.npu_fused_infer_attention_score``, which exposes
        the BlasST ``sparse_lambda`` parameter directly instead of encoding it
        via ``antiquant_mode``.
        """
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(
            key, value, attn_metadata
        )

        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]
        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]

        # The custom op expects int64 device tensors for sequence lengths.
        actual_seq_lengths_q = torch.tensor(
            attn_metadata.actual_seq_lengths_q,
            dtype=torch.int64,
            device=query.device,
        )
        if not isinstance(actual_seq_lengths_kv, torch.Tensor):
            actual_seq_lengths_kv = torch.tensor(
                actual_seq_lengths_kv,
                dtype=torch.int64,
                device=query.device,
            )

        # Match the baseline (torch_npu) path: always sparse_mode=3 with attn_mask.
        sparse_mode = 3
        atten_mask = attn_metadata.attn_mask

        # Sparse lambda is passed directly; -99.0 means "dense" (no skipping).
        sparse_lambda = -99.0
        enable_lse_flag = False  # A/B 结论: lse=True 仍 NaN(occ11)且触发 ScatterElements 崩溃, 回退
        attn_output, softmax_lse = torch.ops._C_ascend.npu_fused_infer_attention_score(
            query,
            key,
            value,
            None,  # pse_shift
            atten_mask,
            actual_seq_lengths_q,
            actual_seq_lengths_kv,
            block_table,
            self.num_heads,
            self.scale,
            self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
            2147483647,  # next_tokens
            "TND",
            self.num_kv_heads,
            sparse_mode,
            0,  # inner_precise
            block_size,
            0,  # antiquant_mode
            sparse_lambda,
            enable_lse_flag,
        )

        stage = _get_stage_name(attn_metadata.attn_state)
        if enable_lse_flag:
            sparse_int, block_int = _parse_glse_block_info(
                softmax_lse,
                layer_name,
                stage,
                query=query,
                key=key,
                value=value,
                attn_output=attn_output,
                block_table=block_table,
                block_size=block_size,
                actual_seq_qlen=attn_metadata.actual_seq_lengths_q,
                actual_seq_kvlen=actual_seq_lengths_kv,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                scale=self.scale,
            )

        if _FIA_DUMP_ON_NAN:
            import re as _re
            _lm = _re.search(r"layers\.(\d+)", layer_name)
            _lid = int(_lm.group(1)) if _lm else -1
            if _lid not in _fia_nan_dumped_layers \
                    and bool(torch.isnan(attn_output).any().item()):
                _fia_nan_dumped_layers.add(_lid)
                print(f"[FIA_DUMP][NAN-TRAP] layer={layer_name} stage={stage} "
                      f"num_tokens={num_tokens} "
                      f"query_nan={bool(torch.isnan(query).any().item())} "
                      f"key_nan={bool(torch.isnan(key).any().item())}",
                      flush=True)
                _fia_repro_dump(
                    layer_name=layer_name,
                    stage=stage,
                    num_tokens=num_tokens,
                    attn_output=attn_output,
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=atten_mask,
                    actual_seq_lengths_q=actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    block_table=block_table,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    scale=self.scale,
                    sparse_mode=sparse_mode,
                    pre_tokens=self.sliding_window
                    if self.sliding_window is not None else SWA_INT_MAX,
                    next_tokens=2147483647,
                    block_size=block_size,
                    sparse_lambda=sparse_lambda,
                    enable_lse_flag=enable_lse_flag,
                    force=True,
                    tag="_nan",
                )

        _fia_repro_dump(
            layer_name=layer_name,
            stage=stage,
            num_tokens=num_tokens,
            attn_output=attn_output,
            query=query,
            key=key,
            value=value,
            atten_mask=atten_mask,
            actual_seq_lengths_q=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            block_table=block_table,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            scale=self.scale,
            sparse_mode=sparse_mode,
            pre_tokens=self.sliding_window
            if self.sliding_window is not None else SWA_INT_MAX,
            next_tokens=2147483647,
            block_size=block_size,
            sparse_lambda=sparse_lambda,
            enable_lse_flag=enable_lse_flag,
        )

        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer_name: str = "",
    ):
        # we inherit ForwardContext in model runner v2, when enable model
        # runner v2, there is not capturing attribute in forward_context,
        # just use getattr to avoid attribute error.
        if _EXTRA_CTX.capturing:
            attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer_name)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and self.sliding_window is not None
            and attn_metadata.seq_lens.shape[0] == query.size(0)
            and self.sinks is None
        ):
            return self._forward_fia_slidingwindow(query, attn_metadata, output)
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]
        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]
        # Get workspace from cache or calculate it if not present.
        if self.sinks is not None:
            actual_seq_qlen = attn_metadata.actual_seq_lengths_q
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                actual_seq_qlen = torch.tensor([1] * len(attn_metadata.seq_lens_list), dtype=torch.int32).cumsum(dim=0)
            if self.sliding_window is not None:
                atten_mask = attn_metadata.swa_mask
                sparse_mode = 4
            else:
                atten_mask = attn_metadata.attn_mask
                sparse_mode = 3
            attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query,
                key,
                value,
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                next_tokens=0,
                atten_mask=atten_mask,
                sparse_mode=sparse_mode,
                softmax_scale=self.scale,
                block_table=block_table,
                block_size=block_size,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_lengths_kv,
                learnable_sink=self.sinks,
            )
        else:
            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )

            attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _EXTRA_CTX.capturing:
            return self.full_graph_pa(query, attn_metadata, output)
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
        )
        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        _: torch.Tensor,
    ) -> torch.Tensor:
        # use default sparse_mode 0 in normal scenario, which means no mask works on it
        return torch_npu.npu_fusion_attention(
            query=query,
            key=key,
            value=value,
            head_num=self.num_heads,
            input_layout="TND",
            scale=self.scale,
            actual_seq_qlen=attn_metadata.actual_seq_lengths_q,
            actual_seq_kvlen=attn_metadata.actual_seq_lengths_q,
        )[0]

    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) > 1:
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event = torch.npu.Event()
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping
            encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER
            DeviceOperator.reshape_and_cache(
                key=key[: attn_metadata.num_actual_tokens] if not encoder_decoder else key,
                value=value[: attn_metadata.num_actual_tokens] if not encoder_decoder else value,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                # quick fix to make sure slots is int32 for cross attention case.
                # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
                slot_mapping=slots[: attn_metadata.num_actual_tokens] if not encoder_decoder else slots.to(torch.int32),
            )
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()
        return query, key, value, output

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer_name: str = "",
    ):
        num_tokens = query.shape[0]
        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and using_paged_attention(num_tokens, self.vllm_config)
            and self.sliding_window is None
        ):
            if layer_name not in _fia_branch_logged_layers:
                print(f"[FIA_BRANCH] layer={layer_name} use forward_paged_attention, "
                      f"attn_state={attn_metadata.attn_state}, num_tokens={num_tokens}", flush=True)
                _fia_branch_logged_layers.add(layer_name)
            output = self.forward_paged_attention(query, attn_metadata, output)
        elif self._can_use_custom_fia(attn_metadata):
            if layer_name not in _fia_branch_logged_layers:
                print(f"[FIA_BRANCH] layer={layer_name} use forward_custom_fused_infer_attention (custom FIA), "
                      f"attn_state={attn_metadata.attn_state}, num_tokens={num_tokens}", flush=True)
                _fia_branch_logged_layers.add(layer_name)
            if _FIA_AB_COMPARE:
                nt = attn_metadata.actual_seq_lengths_q[-1]
                stage = _get_stage_name(attn_metadata.attn_state)
                # 分场景判定：decode-only / 混合 batch / 纯 chunk。
                # raw dump 必须在跑 custom 之前完成 —— 既防设备崩溃丢现场，
                # 又避免 custom 潜在越界写污染输入快照。
                _scen = None
                if stage == "decode":
                    _scen = "decode"
                elif stage == "chunked_prefill":
                    _aqs = attn_metadata.actual_seq_lengths_q
                    _ql = ([_aqs[0]] + [_aqs[i] - _aqs[i - 1]
                                        for i in range(1, len(_aqs))]) if _aqs else []
                    if _ql:
                        _scen = "mixed" if (min(_ql) == 1 and max(_ql) > 1) else "chunk"
                _grab_this = bool(_FIA_AB_STAGE_DUMP and _scen
                                  and _scen not in _fia_stage_dumped)
                if _grab_this:
                    _fia_stage_dumped.add(_scen)
                    print(f"[FIA_AB][STAGE-DUMP] scenario={_scen} layer={layer_name} "
                          f"tokens={nt} (raw, pre-op)", flush=True)
                    try:
                        _k2, _v2, _bs2, _bt2, _akv2 = self._get_fia_params(
                            key, value, attn_metadata)
                        _aq2 = torch.tensor(attn_metadata.actual_seq_lengths_q,
                                            dtype=torch.int64, device=query.device)
                        if not isinstance(_akv2, torch.Tensor):
                            _akv2 = torch.tensor(_akv2, dtype=torch.int64,
                                                 device=query.device)
                        _fia_repro_dump(
                            layer_name=layer_name, stage=stage, num_tokens=nt,
                            attn_output=None, query=query[:nt],
                            key=_k2, value=_v2,
                            atten_mask=attn_metadata.attn_mask,
                            actual_seq_lengths_q=_aq2,
                            actual_seq_lengths_kv=_akv2,
                            block_table=_bt2,
                            num_heads=self.num_heads,
                            num_kv_heads=self.num_kv_heads,
                            scale=self.scale, sparse_mode=3,
                            pre_tokens=SWA_INT_MAX, next_tokens=2147483647,
                            block_size=_bs2, sparse_lambda=-99.0,
                            enable_lse_flag=False,
                            force=True, tag=f"_{_scen}", raw=_FIA_DUMP_RAW)
                    except Exception as _e:
                        print(f"[FIA_AB][STAGE-DUMP] FAILED: {_e}", flush=True)
                # 混合 batch 默认不线上跑 custom（曾致设备崩溃）：baseline 前传，
                # custom 行为靠离线单算子复现。
                if _scen == "mixed" and not _FIA_AB_MIXED_RUN_CUSTOM:
                    output = self.forward_fused_infer_attention(
                        query, key, value, attn_metadata, output, layer_name)
                    print(f"[FIA_AB][MIXED-SKIP] layer={layer_name} tokens={nt} "
                          f"custom 未线上跑，仅 baseline 前传", flush=True)
                    return output
                # A/B 双跑：默认 custom -> 临时 buffer、baseline -> 正式 output 前传；
                # 角色反转按调用判定：仅当前场景列入 _FIA_CUSTOM_STAGES 时
                # custom -> 正式 output 前传，baseline -> 临时 buffer 仅做对比。
                _custom_real = (_FIA_AB_CUSTOM_REAL and _scen is not None
                                and _scen in _FIA_CUSTOM_STAGES)
                if _custom_real:
                    output = self.forward_custom_fused_infer_attention(
                        query, key, value, attn_metadata, output, layer_name)
                    output_custom = output
                    output_base = self.forward_fused_infer_attention(
                        query, key, value, attn_metadata,
                        torch.empty_like(output), layer_name)
                else:
                    output_custom = self.forward_custom_fused_infer_attention(
                        query, key, value, attn_metadata,
                        torch.empty_like(output), layer_name)
                    output_base = None
                    output = self.forward_fused_infer_attention(
                        query, key, value, attn_metadata, output, layer_name)
                oc = output_custom[:nt]
                ob = output_base[:nt] if _custom_real else output[:nt]
                # 双输出快照（对应 pre-op raw dump 的同一次调用）
                if _grab_this:
                    try:
                        torch.save(
                            {"output_custom": oc.detach().cpu(),
                             "output_base": ob.detach().cpu()},
                            os.path.join(
                                _FIA_DUMP_DIR,
                                f"fia_stage_outputs_{_scen}_pid{os.getpid()}_"
                                f"{layer_name.replace('.', '_')}.pt"))
                    except Exception as _e:
                        print(f"[FIA_AB][STAGE-DUMP] outputs save FAILED: {_e}",
                              flush=True)
                nan_c_t = torch.isnan(oc)
                nan_b_t = torch.isnan(ob)
                st = _fia_ab_stats.setdefault(stage, dict(
                    calls=0, nan_calls=0, elems=0, diff=0,
                    max_abs=0.0, max_rel=0.0, hist=[0] * 6,
                    worst=(0.0, "", 0)))
                st["calls"] += 1
                if nan_c_t.any().item() or nan_b_t.any().item():
                    st["nan_calls"] += 1
                    nc = int(nan_c_t.sum().item())
                    nb = int(nan_b_t.sum().item())
                    md = (oc.float() - ob.float()).abs().nan_to_num(0).max().item()
                    same_pos = bool((nan_c_t == nan_b_t).all().item())
                    _fia_ab_nan_events.append((layer_name, stage, nt, nc, nb, md))
                    print(f"[FIA_AB][NAN] layer={layer_name} stage={stage} tokens={nt} "
                          f"nan_custom={nc} nan_base={nb} 位置一致={same_pos} "
                          f"maxdiff(有限)={md:.4e}", flush=True)
                else:
                    # 数值对比：bit 一致率 / abs / rel / ulp 直方图
                    ocf, obf = oc.float(), ob.float()
                    d = (ocf - obf).abs()
                    bitdiff = (oc.view(torch.int16) != ob.view(torch.int16))
                    nd = int(bitdiff.sum().item())
                    st["elems"] += d.numel()
                    st["diff"] += nd
                    if nd:
                        mabs = d.max().item()
                        m = obf.abs() > 1e-3
                        mrel = (d[m] / obf[m].abs()).max().item() if m.any().item() else 0.0
                        ulp = obf.abs() * (2 ** -8) + 1e-30
                        ud = (d / ulp)[bitdiff].long().clamp(0, 5)
                        hist = torch.bincount(ud, minlength=6).tolist()
                        st["hist"] = [a + b for a, b in zip(st["hist"], hist)]
                        st["max_abs"] = max(st["max_abs"], mabs)
                        st["max_rel"] = max(st["max_rel"], mrel)
                        if mrel > st["worst"][0]:
                            st["worst"] = (mrel, layer_name, nt)
                        if mrel > _FIA_AB_REL_ALERT:
                            print(f"[FIA_AB][DIFF] layer={layer_name} stage={stage} tokens={nt} "
                                  f"diff_elems={nd}/{d.numel()} max_abs={mabs:.4e} max_rel={mrel:.4e} "
                                  f"ulp_hist[<1,1,2,3,4,>=5]={hist}", flush=True)
                        # 大差异陷阱：抓 decode/chunk 大差异 batch 的完整算子输入做离线复现
                        global _fia_bigdiff_dump_count
                        if (_FIA_AB_BIGDIFF_DUMP and stage in ("decode", "chunked_prefill")
                                and mabs > _FIA_AB_BIGDIFF_THRESH
                                and _fia_bigdiff_dump_count < _FIA_AB_BIGDIFF_MAX):
                            _fia_bigdiff_dump_count += 1
                            print(f"[FIA_AB][BIGDIFF-DUMP] layer={layer_name} tokens={nt} "
                                  f"max_abs={mabs:.4e} (#{_fia_bigdiff_dump_count})", flush=True)
                            try:
                                _k2, _v2, _bs2, _bt2, _akv2 = self._get_fia_params(
                                    key, value, attn_metadata)
                                _aq2 = torch.tensor(attn_metadata.actual_seq_lengths_q,
                                                    dtype=torch.int64, device=query.device)
                                if not isinstance(_akv2, torch.Tensor):
                                    _akv2 = torch.tensor(_akv2, dtype=torch.int64,
                                                         device=query.device)
                                _fia_repro_dump(
                                    layer_name=layer_name, stage=stage, num_tokens=nt,
                                    attn_output=output_custom[:nt], query=query[:nt],
                                    key=_k2, value=_v2,
                                    atten_mask=attn_metadata.attn_mask,
                                    actual_seq_lengths_q=_aq2,
                                    actual_seq_lengths_kv=_akv2,
                                    block_table=_bt2,
                                    num_heads=self.num_heads,
                                    num_kv_heads=self.num_kv_heads,
                                    scale=self.scale, sparse_mode=3,
                                    pre_tokens=SWA_INT_MAX, next_tokens=2147483647,
                                    block_size=_bs2, sparse_lambda=-99.0,
                                    enable_lse_flag=False,
                                    force=True, tag="_bigdiff", raw=_FIA_DUMP_RAW)
                                torch.save(
                                    {"output_custom": oc.detach().cpu(),
                                     "output_base": ob.detach().cpu()},
                                    os.path.join(
                                        _FIA_DUMP_DIR,
                                        f"fia_bigdiff_outputs_pid{os.getpid()}_"
                                        f"{layer_name.replace('.', '_')}.pt"))
                            except Exception as _e:
                                print(f"[FIA_AB][BIGDIFF-DUMP] FAILED: {_e}", flush=True)
                if st["calls"] % _FIA_AB_STAT_EVERY == 0:
                    exact = 1.0 - st["diff"] / max(st["elems"], 1)
                    w = st["worst"]
                    print(f"[FIA_AB][STAT] stage={stage} calls={st['calls']} nan_calls={st['nan_calls']} "
                          f"bit一致率={exact * 100:.4f}% diff_elems={st['diff']}/{st['elems']} "
                          f"max_abs={st['max_abs']:.4e} max_rel={st['max_rel']:.4e} "
                          f"ulp_hist[<1,1,2,3,4,>=5]={st['hist']} "
                          f"worst=(rel={w[0]:.4e} layer={w[1]} tokens={w[2]})", flush=True)
            else:
                output = self.forward_custom_fused_infer_attention(query, key, value, attn_metadata, output, layer_name)
        else:
            if layer_name not in _fia_branch_logged_layers:
                print(f"[FIA_BRANCH] layer={layer_name} use forward_fused_infer_attention (original FIA), "
                      f"attn_state={attn_metadata.attn_state}, num_tokens={num_tokens}", flush=True)
                _fia_branch_logged_layers.add(layer_name)
            output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output, layer_name)

        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendAttentionBackendImpl")

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)
        output_padded = None
        if key is not None and value is not None:
            output_padded = output
            query, key, value, output_padded = self.reshape_and_cache(
                query, key, value, kv_cache, attn_metadata, output
            )
        # pooling model branch
        if attn_metadata.model_runner_type == "pooling" and not attn_metadata.causal:
            attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        if output_padded is not None:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded, layer.layer_name)
        else:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output, layer.layer_name)
        output[:num_tokens] = attn_output[:num_tokens]
        return output


class AscendC8AttentionBackendImpl(AscendAttentionBackendImpl):
    """Attention backend implementation for INT8 KV cache (C8/QuaRot) models.

    This subclass handles static per-channel INT8 KV cache quantization.
    It is activated via class surgery in AscendC8KVCacheAttentionMethod.create_weights
    (vllm_ascend/quantization/methods/kv_c8.py)
    so that C8 attention layers automatically use this forward path.
    """

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendC8AttentionBackendImpl")

        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        float_key, float_value = None, None
        if key is not None and value is not None:
            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                float_key, float_value = key, value
            key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
            query, key, value, _ = self.reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)

        if attn_metadata.model_runner_type == "pooling":
            attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output

        self._prepare_c8_scales(layer, query.device)
        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            return self._forward_c8_decode(query, attn_metadata, output, layer)
        elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
            return self._forward_c8_chunked_prefill(query, float_key, float_value, attn_metadata, output, layer)
        else:
            return self._forward_c8_fused_infer_attention(
                query,
                float_key if float_key is not None else key,
                float_value if float_value is not None else value,
                attn_metadata,
                output,
                layer,
            )

    def _prepare_c8_scales(self, layer: AttentionLayer, device: torch.device) -> None:
        """Shard per-channel C8 scales/offsets to this TP rank and pre-compute
        BF16 BNSD antiquant tensors for FIA V1 decode fast path.
        """
        if hasattr(layer, "_c8_scales_prepared"):
            return

        def _shard_and_reshape(raw: torch.Tensor) -> torch.Tensor:
            if raw.numel() == 1:
                return raw.to(device=device)
            expected = self.num_kv_heads * self.head_size
            if raw.numel() != expected:
                total_kv_heads = raw.numel() // self.head_size
                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()
                kv_head_start = tp_rank * total_kv_heads // tp_size
                raw = raw.view(total_kv_heads, self.head_size)[
                    kv_head_start : kv_head_start + self.num_kv_heads
                ].contiguous()
            return raw.view(1, self.num_kv_heads, self.head_size).to(device=device)

        layer._c8_k_scale = _shard_and_reshape(layer.k_cache_scale.data)
        layer._c8_k_offset = _shard_and_reshape(layer.k_cache_offset.data)
        layer._c8_v_scale = _shard_and_reshape(layer.v_cache_scale.data)
        layer._c8_v_offset = _shard_and_reshape(layer.v_cache_offset.data)

        bnsd = (1, self.num_kv_heads, 1, self.head_size)
        layer._c8_k_aq_scale = layer._c8_k_scale.to(torch.bfloat16).view(bnsd).contiguous()
        layer._c8_k_aq_offset = layer._c8_k_offset.to(torch.bfloat16).view(bnsd).contiguous()
        layer._c8_v_aq_scale = layer._c8_v_scale.to(torch.bfloat16).view(bnsd).contiguous()
        layer._c8_v_aq_offset = layer._c8_v_offset.to(torch.bfloat16).view(bnsd).contiguous()

        layer._c8_k_inv_scale_bf16 = (1.0 / layer._c8_k_scale).to(torch.bfloat16)
        layer._c8_k_offset_bf16 = layer._c8_k_offset.to(torch.bfloat16)
        layer._c8_v_inv_scale_bf16 = (1.0 / layer._c8_v_scale).to(torch.bfloat16)
        layer._c8_v_offset_bf16 = layer._c8_v_offset.to(torch.bfloat16)

        layer._c8_scales_prepared = True

    def _dequant_paged_kv_to_dense(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: list,
        target_dtype: torch.dtype,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather paged INT8 KV blocks and dequantize to target_dtype."""
        batch_size = block_table.shape[0]
        block_size = key.shape[1]
        H = key.shape[2]
        max_blocks_per_seq = block_table.shape[1]
        max_tokens_padded = max_blocks_per_seq * block_size

        flat_ids = block_table.reshape(-1)
        gathered_k = key[flat_ids].view(batch_size, max_tokens_padded, H)
        gathered_v = value[flat_ids].view(batch_size, max_tokens_padded, H)

        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=key.device)
        positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key.device)
        valid_mask = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).view(-1)

        dense_k = gathered_k.view(-1, H)[valid_mask]
        dense_v = gathered_v.view(-1, H)[valid_mask]

        dense_k = dense_k.view(-1, self.num_kv_heads, self.head_size)
        dense_v = dense_v.view(-1, self.num_kv_heads, self.head_size)
        k_scale = layer._c8_k_scale.to(target_dtype)
        k_offset = layer._c8_k_offset.to(target_dtype)
        v_scale = layer._c8_v_scale.to(target_dtype)
        v_offset = layer._c8_v_offset.to(target_dtype)
        dense_k = (dense_k.to(target_dtype) - k_offset) * k_scale
        dense_v = (dense_v.to(target_dtype) - v_offset) * v_scale
        return dense_k, dense_v

    def _quantize_kv_to_int8(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: AttentionLayer,
        num_actual_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize K/V from float to INT8 using static per-channel C8 scales."""
        self._prepare_c8_scales(layer, key.device)

        actual_key = key[:num_actual_tokens]
        actual_value = value[:num_actual_tokens]

        k_int8 = torch.clamp(
            torch.round(actual_key * layer._c8_k_inv_scale_bf16 + layer._c8_k_offset_bf16),
            -128,
            127,
        ).to(torch.int8)
        v_int8 = torch.clamp(
            torch.round(actual_value * layer._c8_v_inv_scale_bf16 + layer._c8_v_offset_bf16),
            -128,
            127,
        ).to(torch.int8)
        return k_int8, v_int8

    def _forward_c8_decode(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 decode via FIA V1 BNSD with native paged INT8 KV + perchannel antiquant."""
        num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
        assert block_size % 32 == 0, f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
        key = self.key_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]
        value = self.value_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]
        batch_size = len(attn_metadata.seq_lens_list)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query[:batch_size].unsqueeze(2),
            key,
            value,
            key_antiquant_scale=layer._c8_k_aq_scale,
            key_antiquant_offset=layer._c8_k_aq_offset,
            value_antiquant_scale=layer._c8_v_aq_scale,
            value_antiquant_offset=layer._c8_v_aq_offset,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            scale=self.scale,
            block_size=block_size,
            key_antiquant_mode=0,
            value_antiquant_mode=0,
            sparse_mode=0,
        )
        attn_output = attn_output.squeeze(2)
        output[:batch_size] = attn_output
        return output

    def _forward_c8_chunked_prefill(
        self,
        query: torch.Tensor,
        float_key: torch.Tensor | None,
        float_value: torch.Tensor | None,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 ChunkedPrefill: decode via FIA V1 BNSD paged INT8 (zero gather),
        prefill via FIA V1 TND with float KV (new) or gather+dequant (continuing).
        """
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]

        if num_decode_tokens > 0:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
            assert block_size % 32 == 0, (
                f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
            )
            kv_k = self.key_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]
            kv_v = self.value_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]

            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query[:num_decode_tokens].unsqueeze(2),
                kv_k,
                kv_v,
                key_antiquant_scale=layer._c8_k_aq_scale,
                key_antiquant_offset=layer._c8_k_aq_offset,
                value_antiquant_scale=layer._c8_v_aq_scale,
                value_antiquant_offset=layer._c8_v_aq_offset,
                block_table=attn_metadata.block_tables[:num_decodes],
                actual_seq_lengths_kv=attn_metadata.seq_lens_list[:num_decodes],
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD",
                scale=self.scale,
                block_size=block_size,
                key_antiquant_mode=0,
                value_antiquant_mode=0,
                sparse_mode=0,
            )
            output[:num_decode_tokens] = attn_out.squeeze(2)

        if attn_metadata.num_prefills > 0:
            prefill_q = query[num_decode_tokens:num_tokens]

            prefill_seq_qlen = [
                actual_seq_qlen[i] - num_decode_tokens for i in range(num_decodes, len(actual_seq_qlen))
            ]

            all_new_prefill = True
            for i in range(num_decodes, len(attn_metadata.seq_lens_list)):
                q_start = actual_seq_qlen[i - 1] if i > 0 else 0
                qlen_i = actual_seq_qlen[i] - q_start
                if attn_metadata.seq_lens_list[i] > qlen_i:
                    all_new_prefill = False
                    break

            if all_new_prefill and float_key is not None and float_value is not None:
                prefill_k = float_key[num_decode_tokens:num_tokens]
                prefill_v = float_value[num_decode_tokens:num_tokens]
                prefill_seq_kvlen = prefill_seq_qlen
            else:
                num_block, blk_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
                paged_k = self.key_cache.view(num_block, blk_size, -1)  # type: ignore[attr-defined]
                paged_v = self.value_cache.view(num_block, blk_size, -1)  # type: ignore[attr-defined]
                prefill_bt = attn_metadata.block_tables[num_decodes:]
                prefill_sl = attn_metadata.seq_lens_list[num_decodes:]
                prefill_k, prefill_v = self._dequant_paged_kv_to_dense(
                    paged_k, paged_v, prefill_bt, prefill_sl, query.dtype, layer
                )
                prefill_seq_kvlen = torch.tensor(prefill_sl, dtype=torch.int32).cumsum(dim=0)

            # block_table is None for prefill; FIA ignores block_size in this case.
            # Use cache block_size for consistency rather than a magic number.
            cache_block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query=prefill_q,
                key=prefill_k,
                value=prefill_v,
                atten_mask=attn_metadata.attn_mask,
                block_table=None,
                input_layout="TND",
                block_size=cache_block_size,
                actual_seq_lengths=prefill_seq_qlen,
                actual_seq_lengths_kv=prefill_seq_kvlen,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )
            n_prefill = num_tokens - num_decode_tokens
            attn_out = attn_out.view(n_prefill, self.num_heads, self.head_size)
            output[num_decode_tokens:num_tokens] = attn_out[:n_prefill]

        return output

    def _forward_c8_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ):
        """C8 FIA V1 TND for prefill states (PrefillNoCache uses float KV directly,
        PrefillCacheHit gathers + dequants paged INT8 KV).
        """
        self._prepare_c8_scales(layer, query.device)
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)

        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]
        query = query[:num_tokens]

        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]

        if key.dtype == torch.int8:
            if block_table is not None:
                seq_lens = (
                    actual_seq_lengths_kv if isinstance(actual_seq_lengths_kv, list) else actual_seq_lengths_kv.tolist()
                )
                key, value = self._dequant_paged_kv_to_dense(key, value, block_table, seq_lens, query.dtype, layer)
                block_table = None
                # block_table is None after dequant; FIA ignores block_size.
                # Use cache block_size for consistency rather than a magic number.
                block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
                actual_seq_lengths_kv = torch.tensor(seq_lens, dtype=torch.int32).cumsum(dim=0)
            else:
                qdt = query.dtype
                k_scale = layer._c8_k_scale.to(qdt)
                k_offset = layer._c8_k_offset.to(qdt)
                v_scale = layer._c8_v_scale.to(qdt)
                v_offset = layer._c8_v_offset.to(qdt)
                key = (key.to(qdt) - k_offset) * k_scale
                value = (value.to(qdt) - v_offset) * v_scale

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output
        return output
