# BlasST（custom fused_infer_attention_score）代码检视报告

- **检视范围**：commit `f2989fa88`（custom FIA BlasST op: CANN kernel + integration，65 文件 +16301 行）+ 检视开始时工作区的 debug cycle dump 改动（5 文件 +2 未跟踪）
- **检视基准**：vLLM Ascend 合入规范 12 条 + 多角度代码检视（8 个发现角度 + 6 个对抗验证子代理，2 项候选被否证剔除）
- **检视日期**：2026-09-07
- **总体结论**：**当前状态不满足合入标准，不建议合入**。除规范符合性缺口（零 CI 看护、环境变量未评审、魔数/死代码）外，多角度检视确认了 **6 项正确性缺陷**（含 2 项静默错误输出、1 项服务中崩溃）与 2 项移植性越界隐患。

> **检视期间工作区变动说明**：检视进行中，debug cycle dump 的 4 个源文件改动（torch_adpt.h / proto.cpp / kernel.h / attention_v1.py）被移出 `feature/custom-fia` 工作区；reflog 显示并行会话在 `refactor/fia-review-fixes` 分支活动，且 22:13 已产出 `cycle_dump_results_20260907.txt`（逐核计数实测成功）。锚定在已提交 diff 的发现全部仍然有效；唯一仅存在于工作区改动的发现（#8）已单独标注，其在该改动重落地时仍适用。

---

## 一、逐条规范对照

### 规范 1：新增环境变量需要评审 —— ❌ 违规

生产代码（tiling host 侧）新增 2 个未评审、未文档化的环境变量开关：

| 环境变量 | 位置 | 语义 |
|---|---|---|
| `VLLM_FIA_HOST_SEQ_TILING` | `csrc/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp:558` | 默认开启，设 `0` 关闭 host seq tiling 路径 |
| `VLLM_FIA_FD` | 同文件 `:567` | 默认开启，设 `0` 关闭 FlashDecode |

问题：
- 两个变量控制**核心行为**（tiling 策略 / FD 通路），以"默认开、环境变量关闭"的 kill-switch 形式藏在算子里，docs/ 与 README 零提及，PR 描述中也未列出。
- **违反 AGENTS.md 约定**：环境变量绕过了 `envs.py` 集中管理规则（多角度检视的规范核对角度独立发现同一问题）。
- **建议**：评审后并入 `custom_fia_config`（已有 pydantic 配置通道），或按 envs.py 集中登记并文档化。

次要：测试脚本模块级 `os.environ["ASCEND_CUSTOM_OPP_PATH"] = ...`（直接赋值而非 `setdefault`，如 `test_fused_infer_attention_score.py:16`）在 import 时强制覆盖用户环境。

### 规范 2：严禁直接/间接新增模型 —— ✅ 符合

改动集中在 `vllm_ascend/attention/attention_v1.py`（后端分发）、`ascend_config.py`（配置）、`csrc/`（算子），无模型注册/patch。

### 规范 3：严禁拷贝社区类重写，能插件必须插件 —— ⚠️ 基本符合，但有镜像重复

- 未整类拷贝：custom 路径以方法形式加入现有 `AscendAttentionBackendImpl`，门控独立，方式正确。
- 但存在**三组大段镜像代码**（与 baseline `forward_fused_infer_attention` / `full_graph_fia` / update 分支逐行对齐，合计约 300 行）：
  - `forward_custom_fused_infer_attention`（attention_v1.py:1471）
  - `full_graph_custom_fia`（attention_v1.py:1543）
  - update_graph_params 中 `"custom_fia"` 分支（attention_v1.py:835）
- **同一 22 参数算子调用字面量在 3 处复制**（:864、:1515、:1648）；**15 元组靠位置手工对齐**（capture 侧 append :1628-1645 与 update 侧 unpack :841-857）——baseline 一旦改字段顺序，custom 分支静默错位（正确性检视发现 #5/#10 的根因之一）。
- **建议**：抽取公共 kwargs 组装（:1595 的 `ws_args` dict 已证明参数名齐全）；15 元组改 `NamedTuple`/`dataclass` 并按 `PagedAttentionGraphParam` 的方式分发。

### 规范 4：ModelRunner 新增行为重点审视 —— ✅ 无 ModelRunner 改动；注意力分发有多个需敲打的点

新增分发行为 `_can_use_custom_fia`（attention_v1.py:1438）为纯配置驱动（默认关闭），无 `is_xxx` 式模型判断。但多角度检视确认其门控不完备（详见第三节 #2/#3/#5/#6）：

1. **分发优先级与注释不符**：`forward_impl`（:2057-2067）中 `forward_paged_attention` 优先于 custom FIA，DecodeOnly 常规配置下走不到 custom；而注释宣称"全场景单跑 custom"。
2. **缺 causal 检查**：`_can_use_custom_fia` 未检查 `attn_metadata.causal`，非 causal/双向注意力被路由到硬编码 `sparse_mode=3` + mask 的 custom 路径 → 静默错误输出（第三节 #2）。
3. **custom 路径不传 `kv_cache`**：`_get_fia_params(key, value, attn_metadata)`（:1491）省略 baseline 传入的 `kv_cache`，`self.key_cache is None` 时直接 RuntimeError（:1363）而 baseline 可懒初始化（:1353）。

### 规范 5：严禁魔数、随意行为 —— ❌ 违规较多

| 位置 | 魔数 | 说明 |
|---|---|---|
| attention_v1.py:1530, 1602(新) | `3, # sparse_mode` | 两处新代码硬编码；建议提为命名常量/枚举 |
| attention_v1.py:1527, 1663(新) | `2147483647` | 同文件已有 `SWA_INT_MAX`（:71）且同调用内 `pre_tokens` 用了它，`next_tokens` 却写字面量，风格自相矛盾 |
| attention_v1.py:1579(新) | `torch.empty(4096)` | debug 槽位数，无命名/出处注释链接 |
| kernel.h:446-482(工作区改动) | `64 / 2112 / 32 / 4096` | debug 槽位布局魔数，在 kernel.h、proto.cpp 注释、torch_binding_meta.cpp 注释、dump_cycle_stats.py、PROGRESS md **五处重复硬编码**，无共享常量 |
| tiling.cpp:243 附近 | `MAX_CORE_NUM_FD=26` | FD 数组按核数写但只按任务数门控（详见第三节 #7） |
| config/tests 多处 | `-99.0` | DENSE_LAMBDA 语义魔数散落（config 默认值、attention_v1 注释、tests）；经验证 `-99` 哨兵数值上安全（跳过块贡献 <1e-43），但语义应集中定义一次 |

- **死代码**：`_forward_fia_slidingwindow`（attention_v1.py:1408-1436，29 行）全仓无调用点。必须删除。
- **槽位布局无边界保护**：kernel 侧按 `coreIdx` 写槽位，仅注释声明安全范围，无运行时 assert；`dump_cycle_stats.py` 的 `MAX_CORES=30` 同样无断言。详见第三节 #8。

### 规范 6：无 UT/ST 不允许合入 —— ⚠️ 有测试但不可作为合入依据

- 存在 14-case 精度验证与 TEST_DESIGN.md，测试意识好；但测试不在 CI 收集路径且绑定个人机器（见规范 9），对合入把关等同"没有可复现的 UT"。按规范需走韩俊/王玺源豁免，或先改造入 `tests/ut`。
- 且已确认的正确性缺陷多数（非 causal、draft/target 图池、layer-aware replay、混合 batch 数值不变性）恰是当前脚本未覆盖的引擎集成面——**UT 缺口与缺陷分布直接相关**。

### 规范 7：严禁测试不充分直接上库 —— ⚠️ 验证已补做，但缺陷未被现有测试拦截

- 检视期间并行会话完成了 PROGRESS 文档的三步验证（`cycle_dump_results_20260907.txt` 22:13 产出，逐核计数实测成功；`validate_post_cleanup.py` 同步修正了 vendor 路径 `vllm-ascend`→`custom_transformer`）。
- 但下述已确认缺陷（第三节 #1/#2/#3/#5）均为**小规模 UT 覆盖不到的引擎级场景**（>512 并发、双向注意力、投机解码、layer-aware replay），说明现有测试形态对集成路径的看护不足——正是规范 11 要求"解释清楚为何一定用 e2e"的反面佐证。
- 遗留疑点（`vllm_ascend_C.so` mtime 早于提交、`libcust_opapi.so` 时间戳未更新）虽已被后续重建缓解，仍建议在 PR 描述中记录二进制与源码的对应关系。

### 规范 8：py 文件严禁新增全局变量 —— ✅ 生产代码符合

- `attention_v1.py` 新增均为方法/局部变量；`ascend_config.py` 新增 `CustomFIAConfig` 为 pydantic 配置类（合法形态）。
- 测试脚本 `dump_cycle_stats.py` 的常量集合低风险；`validate_post_cleanup.py:35` 的 `_CAUSAL_MASK = None` 模块级可变缓存属测试代码可容忍项，改造入 UT 时建议收敛为 fixture。

### 规范 9：新增特性无 CI 用例禁止合入 —— ❌ 违规（最严重问题）

- CI（`.github/workflows/pr_test.yaml:137,233`）只收集 `tests/ut/**`；本特性全部测试放在 `tests/fused_infer_attention_score/`，**不在任何 CI 收集路径**。
- 测试**无法在 CI/他人机器上运行**：
  - `test_fused_infer_attention_score.py:19-25`：`sys.path.insert(0, "/home/z00603376/ops-transformer-dev/.../tests")` 并 import 外部 golden `row_loop_python_blasst`；
  - 9 个已提交脚本硬编码 `/home/z00603376/blasst_model/kvcomp/...` dump 路径（`perf_driver.py`、`memcheck_driver.py`、`run_dump_repro.py`、`bench_host_vs_device.py`、`gen_sparse_synthetic.py` 等）；
  - 主 UT 依赖的外部 golden 未随仓提供。
- **结论**：custom FIA（3 个 torch op 绑定 + 配置 + 分发逻辑）目前**零 CI 看护**。合入前必须二选一：
  1. 精度 UT 迁入 `tests/ut/vllm_ascend/attention/`，golden 随仓/内联、合成数据去路径化，CI 跑 UT + 粗精度；
  2. 或走 maintainer 强合：label `no-test` + 江少平把关。

### 规范 10：PR 描述写清楚 —— ⚠️ commit message 尚可，PR 化时需补

commit message 有结构，但缺：目标与实测加速比、attention 状态覆盖矩阵与回退场景（PrefillCacheHit/SpecDecoding 回 baseline）、新增环境变量声明、已知限制。另多角度检视指出 **commit message 不符合 AGENTS.md 的 Conventional Commits type/sign-off 约定**。

### 规范 11：优先 UT 覆盖 + 至少粗略精度看护 —— ⚠️ 方向对、形态错

14-case 三级精度对比思路正确，但目前是脚本式，无 pytest 结构/断言化报告/路径解耦。应改造为可 CI 化的 UT；引擎集成面（图模式、spec decode、非 causal）保留少量 e2e 并按规范说明必要性。

### 规范 12：CI 报错与 PR 无关的处理 —— n/a（流程项）

---

## 二、其他检视发现（规范外）

1. **调试产物入库**：f2989fa88 提交了 7 个个人调试/结果文件（`fia_ab_debug_replay_analysis_20260804.txt`、`fia_online_offline_compare.txt`、`precision_cases_findings.md`、`batch_*_summary.txt` ×3、`精度与E2E结果汇总.md`）。应移出（结论沉淀到 PR 描述/TEST_DESIGN.md）。
2. **debug 能力常驻生产 ABI**：proto 将 `sparse_stats` 恒定推导为 `{4096}`，eager 路径每次调用固定分配 16KB（即使 `sparse_stats_flag=False`）。逐核插桩属开发期手段，建议与 release 隔离（flag 关闭时维持 `{2}`，或 debug 编译开关承载）。
3. **热循环插桩非零开销**：flag=false 时仍执行寄存器累加与分支（仅 `GetSystemCycle` 被短路）。"perf path untouched" 表述过强，重落地时应实测确认无回归。
4. **get_workspace 的 Meta 注册返回占位值**（torch_binding_meta.cpp:2128 返回 `query.size(0)`）——torch trace/meta 模式下 workspace 会被 mis-size，图路径仅因实际走真实 op 而未触发。

---

## 三、正确性检视（8 发现角度 + 6 对抗验证，2 项候选被否证剔除）

> 状态说明：#1–#6 经独立验证子代理确认（CONFIRMED，附代码佐证）；#7/#8 为潜在移植性越界（直接读码 + CANN platform_config 核数验证）；被否证的 2 项：`enable_c8_quant` 缺口不可达（C8 后端整体重写 forward）、`sparse_lambda=-99` 哨兵数值上无效差（跳过块贡献 <1e-43）。跨文件契约（Python 调用点 ↔ torch_binding schema ↔ aclnn wrapper ↔ proto/tiling/def.cpp）已追踪，一致。

### CONFIRMED 正确性缺陷

| # | 位置 | 缺陷 | 后果 |
|---|---|---|---|
| 1 | `fused_infer_attention_score_torch_adpt.h:56` | `DevListCache` 容量冻结为 `max(首次n, 512)` 后不再增长，超限直接 `TORCH_CHECK` 而非扩容；`seq_dev_view`（:91，图路径）同缺陷 | `enabled=true` 下首次调用为小 batch（warmup n=8 → cap=512）后，>512 并发请求（v1 `max_num_seqs` 可到 1024）在服务中每次 attention 调用抛 RuntimeError |
| 2 | `attention_v1.py:1438` | `_can_use_custom_fia` 无 causal 检查，而 custom 路径硬编码 `sparse_mode=3` 且恒传 `attn_metadata.attn_mask`（非 causal 时为 None，见 attention_mask.py:58-65；tiling 对 sparse_mode==3 恒映射 MASK_SPEC，tiling.cpp:748） | 双向注意力/交叉注意力被按 causal 掩码 → **静默错误输出**（baseline :1742 用 sparse_mode=0 无 mask）；`full_graph=true` 时 capture 路径 `weak_ref_tensors(None)`（:1635）→ ValueError 崩溃（baseline :1101 有 None guard） |
| 3 | `attention_v1.py:1566` | `full_graph_custom_fia` 忽略 `_EXTRA_CTX.is_draft_model`，恒注册进目标 `get_graph_params()`；baseline（:996-1002）区分 `get_draft_graph_params()` | EAGLE/MTP 投机解码 + `{enabled:true, full_graph:true}`：draft capture 的句柄/事件/参数进主池，draft replay 池查不到 → 录制的 custom-FIA 以 capture 期 seq lens 重放（preload_seq 未执行）→ draft 输出损坏；主池被污染后 update 循环错配 zip → 静默错误 patch |
| 4 | `attention_v1.py:1589` | `graph_params.workspaces[num_tokens]` 与 baseline `full_graph_fia` 共享同一桶，但仅在条目为 None 时定大小，丢弃 baseline 的 `cache_graph_workspace`/`use_max_workspace` 跨算子取大逻辑 | SWA 层（走 baseline）与全局层（走 custom）同 bucket capture（如 Gemma 类混合结构）：先 capture 者定大小，后者复用 → custom 侧 `TORCH_CHECK(ws.numel() >= workspace_size)` 抛错，或 baseline 侧无检查地收欠尺寸 buffer |
| 5 | `attention_v1.py:858` | `"custom_fia"` update 分支用裸 `attn_metadata[key]`，不做 baseline 的 layer_name 感知解析（:929）；capture 元组也不存 layer_name | layer-aware FIA graph replay（gemma4/gemma4_text）+ custom FIA：update 按 `attn_keys[index % num_layers]` 轮询取到别的层的 metadata → 全局层重放拿到 SWA 组的 block_table → **静默错误输出** |
| 6 | `attention_v1.py:1471` | 丢弃 baseline 在 `VLLM_BATCH_INVARIANT` / `CHUNKED_PREFILL_PHASE_SPLIT` 下对混合 decode+prefill batch 的分相处理（:1782-1795） | ChunkedPrefill 混合批（`_can_use_custom_fia` 明确放行）单次融合调用，tiling 的 FD/split-KV 分组随 batch 总组成变化 → 每请求数值随 batch 组成漂移（仓内 test_subbatch_decode.py 已观测到此现象）；且默认栈 `VLLM_BATCH_INVARIANT=1` 时 `_C_ascend` 算子注册被禁 → 该组合下直接未注册算子硬崩溃 |

### 潜在越界（移植隐患）

| # | 位置 | 缺陷 | 后果 |
|---|---|---|---|
| 7 | `fused_infer_attention_score_tiling.cpp:243` | `fillCoreInfoForFlashDecode`/`fillSplitInfoForFlashDecode` 以 `coreIdx < blockNum_ = GetCoreNumAic()` 写 **生产路径**（非 debug 门控）的 `MAX_CORE_NUM_FD=26` 定长数组；FD 门只限 `numTasks<=26`，不限核数 | ≥27 AI 核的 SoC 上 host 侧 tiling 缓冲区越界写；当前唯一注册 SoC ascend910_93 为 20 核（platform_config 已核），安全纯靠巧合 |
| 8 | `fused_infer_attention_score_kernel.h:470`（工作区改动，现已移出本分支） | debug cycle 槽位按 `2112 + coreIdx*64 + subBlockIdx*32` 写 `gSparseStatsOut`，无 coreNum/4096 边界检查（布局仅容纳 ≤31 vec 核）；4096 魔数在 adapter/proto/meta/Python 四处重复 | `sparseStatsFlag=true` 且 ≥32 核时写越 4096 张量末端，静默污染相邻 GM 张量；`dump_cycle_stats.py` 自带 `MAX_CORES=30` 注释承认了 kernel 侧缺失的写入上限。**该改动在 `refactor/fia-review-fixes` 重落地时必须加核数守卫或共享 constexpr** |

### 效率

| # | 位置 | 缺陷 | 后果 |
|---|---|---|---|
| 9 | `fused_infer_attention_score_torch_adpt.h:68` | `upload_seq_lengths` 在 seq 列表变化时对共享 pinned 暂存区 `aclrtSynchronizeEvent` | decode 稳态下 kv 长度每步增长 → 每步首次 custom-FIA 调用（或图 update 前 preload_seq）阻塞 host 至设备排空 → 异步 run-ahead 失效 + 每步一次 host-device 往返。建议：双缓冲 pinned + 双 event / query-then-sync / 旁路流 |

### 简化/复用

| # | 位置 | 缺陷 | 建议 |
|---|---|---|---|
| 10 | `attention_v1.py:1408` | 死方法 `_forward_fia_slidingwindow`（零调用点）；22 参调用字面量三处复制（:864/:1515/:1648）；15 元组位置编码两处对齐（:1628-1645 vs :841-857） | 删除死方法；抽共享 op kwargs 组装（`ws_args` dict 已证明可行）；15 元组改类型化 `CustomFIAGraphParam`，按 `PagedAttentionGraphParam` 方式分发 |

---

## 四、合入前必修清单（按优先级）

**P0 — 正确性缺陷（不修不合入）**

| # | 事项 | 对应发现 |
|---|---|---|
| 1 | 补 `_can_use_custom_fia` 的 causal 门控（非 causal 回退 baseline sparse_mode=0 路径）；图路径补 attn_mask None guard | 三-#2 |
| 2 | `full_graph_custom_fia` 补 `is_draft_model` 分流（draft/target 图参数池隔离） | 三-#3 |
| 3 | update 分支补 layer_name 感知的 metadata 解析，capture 元组带 layer_name | 三-#5 |
| 4 | `DevListCache`/`seq_dev_view` 容量改为可增长（超限扩容而非 TORCH_CHECK） | 三-#1 |
| 5 | workspace 桶与 baseline 的 max-sizing 对齐（复用 `cache_graph_workspace`/`use_max_workspace`） | 三-#4 |
| 6 | 混合 batch 的 phase-split 对齐 baseline，或明确文档化数值不变性取舍并禁用与 `VLLM_BATCH_INVARIANT` 的组合 | 三-#6 |
| 7 | FD tiling 数组写入加核数上界（`coreIdx < MAX_CORE_NUM_FD`），消除 26 数组 vs 核数隐患 | 三-#7 |

**P1 — 规范符合性**

| # | 事项 | 对应规范 |
|---|---|---|
| 8 | 精度 UT 迁入 `tests/ut/`，去除 `/home/z00603376/` 硬编码与外部 golden 依赖，接 CI；或走 no-test 豁免（label + 把关人） | 6/9/11 |
| 9 | `VLLM_FIA_HOST_SEQ_TILING`/`VLLM_FIA_FD` 评审并文档化，按 envs.py 集中登记或并入 custom_fia_config | 1 |
| 10 | 删除死代码 `_forward_fia_slidingwindow`；三处调用字面量抽共享 kwargs；15 元组结构化 | 5/三-#10 |
| 11 | 魔数收敛：sparse_mode/next_tokens 常量化；debug 槽位布局共享 constexpr + 核数守卫（重落地 refactor 分支时） | 5/三-#8 |
| 12 | 修正分发优先级注释或顺序；说明 `_get_fia_params` 不传 kv_cache 的安全性 | 4 |
| 13 | 修 `upload_seq_lengths` 每步同步（双缓冲/旁路流） | 三-#9 |

**P2 — 质量/文档**

| # | 事项 | 对应规范 |
|---|---|---|
| 14 | 移除仓内 7 个个人调试产物文件 | 质量 |
| 15 | 评估 `sparse_stats {4096}` 常驻 ABI 的必要性（flag 关闭时回退 `{2}`）；修 get_workspace Meta 占位返回 | 5/二-2/二-4 |
| 16 | commit/PR 描述规范化：Conventional Commits、性能数据、状态覆盖矩阵、已知限制 | 10 |
