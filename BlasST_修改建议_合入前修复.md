# BlasST（custom FIA）合入前修改建议与修改点

- **基准**：HEAD `f2989fa88` + 当前工作区（并行会话已重落 `{16}` cacheline 修复）
- **配套文档**：`BlasST_代码检视报告_合入规范对照.md`（缺陷详情与规范对照）
- **行号说明**：`attention_v1.py` 行号基于工作区当前内容；`torch_adpt.h` 行号基于已含 `{16}` 修复的当前内容
- **原则**：所有修改建议以"对齐 baseline 已有实现"为第一选择——baseline 的同款逻辑已经过线上验证，镜像它比新写更经得起敲打

## 0. 修改点总览

| # | 修改点 | 文件:位置 | 方案 | 优先级 | 预估 |
|---|---|---|---|---|---|
| 1 | causal 门控 | attention_v1.py:1438 `_can_use_custom_fia` | 非 causal 回退 baseline | P0 | 0.5h |
| 2 | batch-invariant 组合门控 | attention_v1.py:1438 同上 | BATCH_INVARIANT 下禁用 custom | P0 | 0.5h |
| 3 | draft model 门控 | attention_v1.py:1438 同上 | is_draft_model 下禁用（短期） | P0 | 0.5h |
| 4 | layer-aware replay 门控 | attention_v1.py:1452 capture 分支 | layer-aware 模式禁用 capture | P0 | 0.5h |
| 5 | DevListCache 拆分与可增长 | torch_adpt.h:37-95 | eager/graph 缓存分离 + 扩容 | P0 | 4h |
| 6 | workspace 桶 max-sizing | attention_v1.py:1589 | 复用 cache_graph_workspace | P0 | 2h |
| 7 | FD 核数钳位 | tiling.cpp:786-790/:487 | blockNum_ 钳到 MAX_CORE_NUM_FD | P0 | 1h |
| 8 | 混合 batch phase-split | attention_v1.py:1438 | 短期门控，长期镜像 split | P0 | 1h/8h |
| 9 | 三处调用抽公共 kwargs + 结构化参数 | attention_v1.py:864/1515/1648 | 共享 dict + NamedTuple | P1 | 4h |
| 10 | 删死代码 | attention_v1.py:1408-1436 | 删除 | P1 | 0.1h |
| 11 | 环境变量并入配置 | tiling.cpp:558/567 + ascend_config.py | op attr 下发 | P1 | 4h |
| 12 | UT 迁移 tests/ut | tests/fused_infer_attention_score/ → tests/ut/attention/ | pytest 化 + 仓内 golden | P1 | 16h |
| 13 | 魔数收敛 | 多处 | 命名常量 | P1 | 2h |
| 14 | preload 同步优化 | torch_adpt.h:57-60 | eager 双缓冲 | P2 | 4h |
| 15 | tiling 调试日志降级 | tiling.cpp:851 | OPS_LOG_I → OPS_LOG_D | P2 | 0.1h |
| 16 | 移除个人调试产物 | tests/fused_infer_attention_score/7 个文件 | git rm | P2 | 0.5h |
| 17 | 分发优先级注释/顺序 | attention_v1.py:2057-2067 | 调序或改注释 | P2 | 0.5h |
| 18 | `_get_fia_params` 传 kv_cache | attention_v1.py:1491/:1553 | 对齐 baseline | P2 | 0.1h |

---

## 1. P0 正确性修复

### 1.1 causal 门控（缺陷 #2：非 causal 静默错误输出）

**现状**：`_can_use_custom_fia`（attention_v1.py:1438-1469）检查了 sinks/sliding_window/attn_state，但没检查 causal。custom 路径硬编码 `sparse_mode=3` 且恒传 `attn_metadata.attn_mask`——非 causal 时 mask 为 None（attention_mask.py:58-65），tiling 对 sparse_mode==3 恒映射 MASK_SPEC（tiling.cpp:748）→ 双向注意力被按 causal 掩码。

**修改点**：`vllm_ascend/attention/attention_v1.py:1458`（`if self.sinks is not None:` 之前）加：

```python
if not attn_metadata.causal:
    # custom op 仅覆盖 causal（sparse_mode=3 with mask）与 dense 等长场景；
    # 非 causal/交叉注意力回退 baseline sparse_mode=0 路径（见 :1742）
    return False
```

**参照**：baseline `full_graph_fia` 的 sparse_mode 选择（:1007）`sparse_mode = 4 if self.sliding_window else 3 if attn_metadata.causal else 0`——baseline 三分支齐全，custom 只实现了其中一支。
**长期**（可选）：custom op 支持 `sparse_mode=0` 无 mask 路径后放开此门控。
**验证**：UT 加非 causal 用例（bidirectional，对比 baseline 输出 allclose）。

### 1.2 batch-invariant 组合门控（缺陷 #6b：未注册算子硬崩溃）

**现状**：默认栈 `VLLM_BATCH_INVARIANT=1` 时 `_C_ascend` 算子注册被禁用，`custom_fia.enabled=True` 首次 attention 调用 `torch.ops._C_ascend.npu_fused_infer_attention_score` 直接 AttributeError/崩溃。

**修改点**：`_can_use_custom_fia` 函数开头（:1450 `if not fia_config.enabled:` 之前或之后）加：

```python
if envs_vllm.VLLM_BATCH_INVARIANT:
    # batch-invariant 模式下 _C_ascend 自定义算子不注册，且本路径未实现
    # 分相处理（见 baseline :1782-1795 的 phase split），回退 baseline。
    return False
```

（`envs_vllm` 已在该文件 import，baseline :1783 就在用。）

### 1.3 draft model 门控（缺陷 #3：draft/target 图参数池污染）

**现状**：`full_graph_custom_fia`（:1543）不像 baseline `full_graph_fia`（:996-1002）那样区分 `get_draft_graph_params()`/`get_draft_graph_prefill_params()`，EAGLE/MTP 时 draft 捕获写进主池 → 重放错配。

**短期修改点**（推荐，先保正确）：`_can_use_custom_fia` capture 分支（:1452-1457）扩一个条件：

```python
if _EXTRA_CTX.capturing:
    if not (fia_config.full_graph and
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly):
        return False
    if _EXTRA_CTX.is_draft_model:
        # draft 图池分流未实现（baseline :996-1002 有），draft 仍走 baseline
        return False
```

**长期**（若 draft 性能重要）：镜像 baseline 三处——capture 侧 graph_params 选择（:996-1002）、workspace 更新分流（:1084-1087 `update_draft_graph_params_workspaces`）、update 分支 draft 步进（:919-927 `draft_attn_key_steps`）。
**验证**：MTP/EAGLE 模型 e2e（这是必须 e2e 的场景——图捕获语义 UT 无法覆盖，PR 描述里按规范 11 说明）。

### 1.4 layer-aware replay 门控（缺陷 #5：跨层 metadata 错配）

**现状**：`"custom_fia"` update 分支（:858）用裸 `attn_metadata[key]`，不做 baseline 的 layer_name 解析（:929）；capture 元组（:1628-1645）也不存 layer_name。gemma4 类 SWA+全局混合层模型上，全局层重放会拿到 SWA 组的 block_table。

**短期修改点**：`full_graph_custom_fia` 入口（或 `_can_use_custom_fia` capture 分支）加：

```python
if self._use_layer_aware_fia_graph_replay:
    # custom_fia 的 update 分支未实现 layer_name 感知解析（baseline :929），
    # 混合注意力组模型上图重放会跨层错配 metadata，暂不 capture
    return False
```

（`self._use_layer_aware_fia_graph_replay` 已在 `__init__` 初始化：:503。）
**长期**：capture 元组补 `layer_name`（用现成的 `self._graph_metadata_layer_name(layer)`，见 :1124），update 分支镜像 :929 的解析；同时 `full_graph_custom_fia` 需要 `layer` 参数（baseline `full_graph_fia` 有，custom 漏了）。
**验证**：gemma4_text e2e decode 对比 baseline 输出。

### 1.5 DevListCache 拆分与可增长（缺陷 #1：服务中崩溃）

**现状**：`torch_adpt.h` 中 eager（`upload_seq_lengths`，:37-73）与图路径（`seq_dev_view`，:81-95）共用同一对 thread_local 缓存（`seq_len_caches()` :75-79）；容量首次分配定为 `max(n, 512)` 后永不增长，超限 `TORCH_CHECK` 崩溃。且存在隐藏约束冲突：eager 路径可以换地址（executor 每次调用重建），图路径不能（capture 录制了 `cache.dev` 地址，`preload_seq` 原地刷新）——共用缓存时 eager 侧任何扩容都会破坏已捕获图。

**修改点**：`csrc/attention/fused_infer_attention_score/fused_infer_attention_score_torch_adpt.h`

1. **缓存拆分**（地址稳定性隔离）：

```cpp
struct SeqLenCachePair {
    DevListCache eager;    // 地址可变：每次调用重建 executor
    DevListCache graph;    // 地址恒定：capture 录制 + preload 原地刷新
};

inline std::pair<DevListCache *, DevListCache *> seq_len_caches(bool graph)
{
    thread_local SeqLenCachePair *aq = new SeqLenCachePair(),
                                 *akv = new SeqLenCachePair();
    return graph ? std::make_pair(&aq->graph, &akv->graph)
                 : std::make_pair(&aq->eager, &akv->eager);
}
```

调用侧：`fia_exec_common` 按 `no_copy` 选 `seq_len_caches(/*graph=*/no_copy)`（:196-207）；`preload_seq`（:101-103）与 `get_workspace`（:301-305）用 graph 对。

2. **eager 缓存可增长**（替换 :54-56 的 TORCH_CHECK）：

```cpp
if (n > cache.pin.numel()) {
    // eager 路径无地址稳定性约束，直接扩容（2x 避免频繁重分配）
    const int64_t cap = std::max<int64_t>({2 * cache.pin.numel(), n, 512});
    cache.pin = at::empty({cap}, at::TensorOptions().dtype(at::kLong).pinned_memory(true));
    cache.dev = at::empty({cap}, at::TensorOptions().dtype(at::kLong).device(target));
    cache.shadow.clear();
}
```

3. **graph 缓存容量上限**：图路径的 n 恒 ≤ decode bucket 大小 ≤ `max_num_seqs`，捕获更大的 bucket 时才可能超容量——那只发生在 capture 期（安全点，地址变化随 capture 生效）。保留 `seq_dev_view` 的 TORCH_CHECK 作为"replay 期 n 超过捕获容量"这一不变量违反的哨兵即可，但把首次容量从 `max(n, 512)` 提到从 Python 侧传入的 bucket 上界（见第 4 点的 `max_num_seqs`），或至少保留现状并在注释里写明语义。

4. **（可选加固）**：`full_graph_custom_fia` 首次分配 workspace 时已经知道 `self.vllm_config.scheduler_config.max_num_seqs`——把它作为 `preload_seq` 的新可选参数 `min_capacity` 透传给 graph 缓存，一次性分配到位，彻底消除 capture 期扩容。

**验证**：UT 模拟（无 NPU 也行）：容量 512 的缓存灌 600 元素列表不抛错；图路径捕获 bucket 8 后 replay n=8 正常、n=16 报清晰错误。

### 1.6 workspace 桶 max-sizing（缺陷 #4：混合层 capture 崩溃）

**现状**：`full_graph_custom_fia`（:1589-1611）只在 `workspaces.get(num_tokens) is None` 时定大小，与 baseline `full_graph_fia` 共享 `graph_params.workspaces` 桶但丢弃其跨算子取大逻辑（baseline :1032-1061：`use_max_workspace` 时每次 capture 都算 candidate 并经 `cache_graph_workspace` 取 max）。SWA 层（baseline）+ 全局层（custom）同 bucket 时后到者拿到欠尺寸 buffer → custom 侧 TORCH_CHECK 抛错（adapter :141-143）或 baseline 侧无检查用欠尺寸内存。

**修改点**：`vllm_ascend/attention/attention_v1.py:1589` 起的 workspace 段改为复用 baseline 的 helper：

```python
candidate = torch.empty(max(ws_size, ws_size_fd), dtype=torch.uint8, device=query.device)
# 或直接传 size——cache_graph_workspace 的入参形态跟随 baseline :1056-1061
workspace = cache_graph_workspace(
    graph_params, num_tokens, candidate,
    use_max_workspace=self._use_max_workspace_for_fia_graph)
update_graph_params_workspaces(num_tokens, workspace)   # draft 分流见 1.3
```

即：从"None 才分配"改为"每次 capture 都算 candidate，经 `cache_graph_workspace` 与桶内其他算子取 max"（`self._use_max_workspace_for_fia_graph` 在 :504 已初始化，直接用）。`ws_size_fd` 的 FD 上界推导逻辑保留不变。
**验证**：UT——同 bucket 先后以两个不同 workspace 需求调用 `cache_graph_workspace`，断言取大。

### 1.7 FD 核数钳位（缺陷 #7：tiling 数组越界，生产路径）

**现状**：`MAX_CORE_NUM_FD = 26`（tiling.h:29）的定长数组（tiling.h:32-53），写循环按 `coreIdx < blockNum_`（tiling.cpp:243/262/389，blockNum_ 来自 `GetCoreNumAic()` :809）。FD 门控（:787）只限 `numTasks <= 26`，不限核数。当前 ascend910_93 是 20 核所以安全；任何 ≥27 核的 SoC 注册即 host 侧越界写。

**修改点**：`csrc/.../op_host/fused_infer_attention_score_tiling.cpp`——FD 选中分支（:786-790 设 `flashDecodeFlag = true` 处）之后、任务均分（:487 `(totalTaskNum + blockNum_ - 1) / blockNum_`）之前，把 `blockNum_` 钳位：

```cpp
// FD 路径的 tiling 数组按 MAX_CORE_NUM_FD 定长，核数超限时必须钳位，
// 否则 fillCoreInfoForFlashDecode/fillSplitInfoForFlashDecode 越界写。
blockNum_ = std::min(blockNum_, static_cast<uint32_t>(MAX_CORE_NUM_FD));
```

注意必须在 :487 的均分**之前**钳（否则任务被截断丢失）；钳位仅限 FD 分支，非 FD 路径继续用全部核。
**验证**：单元上难以模拟多核 SoC——加 `static_assert(MAX_CORE_NUM_FD >= 1)` 意义不大，改为在 :787 门控处补一条防御性检查 + 注释声明数组容量约束；CI 里至少跑通 FD 用例（decode 长序列）确认 20 核行为不变。

### 1.8 混合 batch phase-split（缺陷 #6a：数值随 batch 组成漂移）

**现状**：`_can_use_custom_fia` 放行 ChunkedPrefill 混合批（decode+prefill 同批），baseline 在 `VLLM_BATCH_INVARIANT` 或 `CHUNKED_PREFILL_PHASE_SPLIT` 能力下会拆相处理（:1782-1795 `_forward_fia_chunked_prefill_split`）保证批不变性；custom 单次融合调用的 tiling 分组随 batch 总组成变化。

**短期修改点**：`_can_use_custom_fia`（:1462 注释宣称支持混合批处）加：

```python
if (attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill
        and attn_metadata.num_decodes > 0
        and attn_metadata.num_prefills > 0
        and (envs_vllm.VLLM_BATCH_INVARIANT
             or get_current_hardware_profile().supports(
                 HardwareCapability.CHUNKED_PREFILL_PHASE_SPLIT))):
    # 混合批分相保证批不变性（baseline :1782-1795）；custom 未实现分相
    return False
```

（导入跟随 :1782-1795 现有用法。）同时修正 :1462 的注释——"含 decode+prefill chunk 混合 batch" 改为准确描述。
**长期**：实现 `_forward_custom_fia_chunked_prefill_split` 镜像 baseline 的拆分逻辑（两个 eager 调用各走 custom op）。
**验证**：UT 固定种子下混合批 vs 分开跑的输出一致性对比（这正是 `test_subbatch_decode.py` 已观测到的漂移，把它变成断言）。

---

## 2. P1 规范符合性

### 2.1 三处调用抽公共 kwargs + 15 元组结构化（缺陷 #10）

**现状**：同一 22 参调用字面量在 :864（update 窗口）、:1515（eager）、:1648（capture）三处复制；15 元组在 append（:1628-1645）与 unpack（:841-857）两处按位置对齐——字段增删/换序会静默错位（1.3/1.4 两个缺陷的共同根因）。

**修改点**：

```python
def _custom_fia_call_kwargs(self, *, sparse_lambda, **seq_overrides):
    """custom FIA 的 op 调用参数（三处调用点共享；字段名即 schema 名）。"""
    return {
        "pse_shift": None,
        "pre_tokens": SWA_INT_MAX,
        "next_tokens": SWA_INT_MAX,
        "input_layout": "TND",
        "sparse_mode": FIA_SPARSE_MODE_COMPRESSED_CAUSAL,   # 见 2.3
        "inner_precise": 0,
        "antiquant_mode": 0,
        "num_heads": self.num_heads,
        "num_key_value_heads": self.num_kv_heads,
        "scale": self.scale,
        "sparse_lambda": sparse_lambda,
        "softmax_lse_flag": False,
        "sparse_stats_flag": False,
        **seq_overrides,
    }
```

三个调用点改为 `torch.ops._C_ascend.npu_fused_infer_attention_score*(query, key, value, atten_mask=..., actual_seq_lengths=..., actual_seq_lengths_kv=..., blocktable=..., **self._custom_fia_call_kwargs(...))`（torch schema 定义了参数名，支持 kwargs）。

元组结构化：

```python
@dataclass
class CustomFIAGraphParam:
    query: ...          # weak ref
    key_cache: ...
    value: ...
    block_table: ...
    attn_mask: ...      # None-safe，见 1.1
    block_size: int
    num_kv_heads: int
    num_heads: int
    scale: float
    attention_out: ...
    softmax_lse: ...
    sparse_stats: ...
```

update 循环按类型分发（对照 `PagedAttentionGraphParam` 的现有模式），淘汰 `"custom_fia"` 字符串标记 + 位置解包。

### 2.2 删除死代码

`_forward_fia_slidingwindow`（attention_v1.py:1408-1436）：全仓零调用，直接删除。

### 2.3 环境变量并入配置（规范 1）

**现状**：`VLLM_FIA_HOST_SEQ_TILING`（tiling.cpp:558）、`VLLM_FIA_FD`（:567）以 getenv kill-switch 形式藏在 host 代码，绕过 `envs.py` 集中管理与 `custom_fia_config` 配置体系。

**修改点**（推荐路径：op attr 下发，kill-switch 能力保留但可评审可测试）：

1. `vllm_ascend/ascend_config.py:119-121` 的 `CustomFIAConfig` 增加字段：

```python
host_seq_tiling: bool = True   # tiling 从 host attrs 读 seq（关闭则走 D2H 回读）
flash_decode: bool = True      # FlashDecode 通路开关
```

2. `csrc/torch_binding.cpp` 三个 op schema 增加对应 `bool` 参数（默认 True）；`fia_exec_common` 透传为 op attr。
3. tiling 侧 `IsHostSeqValueTilingEnabled()`/`IsFlashDecodeEnabled()` 改为读 op attr（`context->GetAttrs()`），删除 getenv。
4. Python 调用点（2.1 的公共 kwargs）从 config 取值传入。

**过渡方案**（若不动 C++ 接口）：至少在 `envs.py` 的 `env_variables` 注册两个变量并在 docs 说明——但 op attr 路径同时解决"评审可见"与"图模式确定性"（getenv 在进程生命周期只读一次 `static`，capture 后改环境无效，是另一个坑），推荐一步到位。

### 2.4 UT 迁移 tests/ut（规范 6/9/11）

**现状**：全部测试在 `tests/fused_infer_attention_score/`（CI 只收集 `tests/ut/**`），主 UT 依赖 `/home/z00603376/ops-transformer-dev` 外部 golden，9 个脚本硬编码个人 dump 路径——CI 与他人机器不可复现。

**修改点**：

1. 新建 `tests/ut/attention/test_custom_fia.py`，模式照抄 `test_attention_fa3.py`（:78 的 skipif + parametrize）：

```python
def _custom_fia_available() -> bool:
    try:
        import torch_npu  # noqa: F401
        return hasattr(torch.ops._C_ascend, "npu_fused_infer_attention_score")
    except ImportError:
        return False

@pytest.mark.skipif(not _custom_fia_available(), reason="custom FIA op not built")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [True])            # 非 causal 门控后走 baseline
@pytest.mark.parametrize("gqa", [False, True])
@pytest.mark.parametrize("kv_lens", [[640, 1408], [2048] * 4])
def test_custom_fia_precision(dtype, causal, gqa, kv_lens): ...
```

2. golden 用**仓内** `blasst_golden_tnd.py`（已随仓提交）替代外部 `optimized_blasst_sim`；精度口径沿用现有三级对比（custom vs golden vs torch_npu），断言 allclose + `stats[0]/stats[1]` 与 golden 跳块数一致。
3. 1.1/1.5/1.6/1.8 的验证用例并入同一文件。
4. `tests/fused_infer_attention_score/` 里**已提交的** 9 个硬编码个人路径脚本与 7 个调试产物文件 `git rm`（下 2.6）；确有价值的（dump_cycle_stats.py 等）留在本地不入仓，或去路径化后挪入 `tools/`。
5. e2e 仅保留图模式/spec-decode 两个 UT 覆盖不了的场景，PR 描述按规范 11 写明理由。

### 2.5 魔数收敛（规范 5）

| 魔数 | 修改点 |
|---|---|
| `3, # sparse_mode`（:1530/:1602 等） | attention_v1.py 模块级 `FIA_SPARSE_MODE_COMPRESSED_CAUSAL = 3`（与文件内既有 SWA_INT_MAX 并列，补注释） |
| `2147483647`（:1527/:1663） | 复用既有 `SWA_INT_MAX`（:71），删字面量 |
| `-99.0` | `ascend_config.py` 侧 `DENSE_LAMBDA: float = -99.0` 类常量 + docstring，attention_v1 注释引用 |
| `sparse_stats {16}`（torch_adpt.h:247/:308、proto.cpp:49、torch_binding_meta.cpp:2072） | torch_adpt.h 定义 `constexpr int64_t kSparseStatsElems = 16;`（含 cacheline 注释），proto/meta 注释引用；四处同步靠注释互指 |
| kernel 侧槽位常量 | 本次工作区未落回；refactor 分支重落地时在 op_kernel 定义共享 constexpr + 核数守卫（检视报告 #8） |

### 2.6 移除个人调试产物

`git rm`（f2989fa88 已提交的 7 个）：`fia_ab_debug_replay_analysis_20260804.txt`、`fia_online_offline_compare.txt`、`precision_cases_findings.md`、`batch_baseline_pre_refactor_summary.txt`、`batch_post_final_summary.txt`、`batch_post_flatten_summary.txt`、`精度与E2E结果汇总.md`。结论性数据沉淀进 PR 描述/TEST_DESIGN.md。工作区未跟踪的 log/prof/`__pycache__` 不入库（可补 .gitignore）。

---

## 3. P2 质量/性能

### 3.1 preload 同步优化（缺陷 #9）

`upload_seq_lengths` 的 `aclrtSynchronizeEvent`（torch_adpt.h:57-60）在 decode 稳态每步阻塞 host。**仅 eager 缓存**做双缓冲（图缓存保持单槽同步，地址稳定性优先）：

```cpp
struct DevListCache {
    ...
    int slot = 0;                       // eager 路径双缓冲轮转
    at::Tensor pin2, dev2;              // 或数组化 pin[2]/dev[2]/ev[2]
};
// 上传前只同步即将复用的那个槽（上一轮早已完成，通常立即返回）
```

落地前提：1.5 的缓存拆分先做。图路径的每步同步待"task update 是否重 patch 地址"验证后再优化。

### 3.2 tiling 调试日志降级

`OPS_LOG_I(... "FIA debug: key=%lu ...")`（tiling.cpp:851）每次算子调用打一条 INFO，生产噪音。改 `OPS_LOG_D` 或删（tiling key 可从 dump 工具获取）。

### 3.3 分发优先级

`forward_impl`（:2057-2067）：paged_attention 优先于 custom FIA，`enabled=True` 时 decode 大多走不到 custom。两个选项：
- **A（推荐）**：`_can_use_custom_fia` 命中时提前——把 custom 分支挪到 paged_attention 之前，注释说明"显式开启即接管 decode"；影响面=已开启用户（当前仅自测）；
- **B**：保持现状，修正 :1462 注释为"DecodeOnly 且未启用 paged attention 时接管"，并在 `CustomFIAConfig` docstring 写明与 paged attention 的优先级关系。

### 3.4 `_get_fia_params` 传 kv_cache

:1491 与 :1553 两处调用补第 4 参 `kv_cache`——`forward_custom_fused_infer_attention`/`full_graph_custom_fia` 需先接收 kv_cache（`forward_impl` 调用处 :2064-2065 传入即可），对齐 baseline 的懒初始化路径（:1353-1361）。

### 3.5 get_workspace Meta 占位返回

`torch_binding_meta.cpp:2128` 的 `-> int` meta 返回 `query.size(0)` 占位：torch.compile trace 下会得到错误 workspace 尺寸。当前调用点（full_graph_custom_fia）不经过 meta，但留着是坑——meta 里加注释声明"仅供 schema 对齐，真实尺寸必须经真实 op 获取"，或在返回处 `TORCH_WARN` 一次性提示。

---

## 4. 落地顺序建议

```
第 1 批（半天，纯 Python 门控，零风险）   1.1 → 1.2 → 1.3 → 1.4 → 2.2 → 3.4 → 3.3(B)
第 2 批（1-2 天，正确性主体）            1.5 → 1.6 → 1.7 → 1.8
第 3 批（结构化，为后续维护兜底）         2.1 → 2.3 → 2.5
第 4 批（测试，合入前必须完成）           2.4（含 1.x 的验证用例）→ 2.6
第 5 批（锦上添花）                      3.1 → 3.2 → 3.5
```

每批完成后的回归：`validate_post_cleanup.py` 14 case 精度 + 一次 decode e2e（custom_fia 开/关各一遍）。

## 5. 已完成项（并行会话，勿重复）

- `sparse_stats {2}→{16}` cacheline 修复（torch_adpt.h:247/:308、proto.cpp:49、torch_binding_meta.cpp:2072）——修的是 kernel 对 stats 张量整行 cacheline 回写踩邻 56 字节的缺陷，与检视报告"flag 关闭时维持小 tensor"建议一致；剩余动作只有 2.5 的常量化。
- `validate_post_cleanup.py:15` vendor 路径修正（`vllm-ascend`→`custom_transformer`）。
- 逐核 cycle 插桩（原 {4096} 版）已从本分支移除；若在 `refactor/fia-review-fixes` 重落地，必须带核数守卫 + 共享 constexpr（检视报告 #8）。
