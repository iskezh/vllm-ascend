# BlasST（custom FIA）全量遗留问题清单

- **日期**：2026-09-08
- **来源**：代码检视报告（12 条合入规范对照 + 10 项正确性检视）+ 修改建议落地过程 + 并行会话协作状态
- **配套文档**：`BlasST_代码检视报告_合入规范对照.md`（缺陷详情）、`BlasST_修改建议_合入前修复.md`（修法与代码草图，本清单的 P0 项均对应其 §1）

---

## 0. 状态快照（先读，最高优先级）

| 项 | 状态 |
|---|---|
| **git** | `feature/custom-fia` HEAD=`1493369b7`；原特性提交 `f2989fa88` 已被并行会话 `reset HEAD~` 撤销，**全部 BlasST 改动（原提交 + {16} 修复 + 环境变量修复 + 测试）都在工作区未提交** |
| **未跟踪目录风险** | `csrc/attention/fused_infer_attention_score/`（全部算子源码）与 `tests/fused_infer_attention_score/` 均为 untracked——**一次误操作 `git clean -fd` 即全部丢失**。建议尽快分批提交 |
| **并行分支** | `refactor/fia-review-fixes`（W2 重构线，tip=671d822a4 "FIA restructure + config-driven refactor"）——reset 意图（重组提交/换基底）需与并行会话确认，避免双方交叉改动 |
| **二进制** | op 包（含新 attr）与 `vllm_ascend_C` 扩展均已重建并安装到 `vendors/custom_transformer`，e2e 16/16 验证通过 |
| **本次已完成** | 环境变量并入 config（9 文件）；`tests/e2e/.../test_custom_fia_precision.py`（16 活跃用例 + 2 skip）+ `test_config.yaml` 注册（180s）；`tests/ut/test_ascend_config.py` + `TestCustomFIAConfig`（2/2）；{16} cacheline 修复（并行会话） |

---

## A. 正确性缺陷（检视 CONFIRMED，**全部未修**——建议随重落地 commit 一起修）

> 修法与代码草图见《修改建议》§1；每项短期方案都是 `_can_use_custom_fia` 里的一行回退门控。

| # | 缺陷 | 位置 | 触发场景 / 后果 | 短期修法 | 长期方案 | 状态 |
|---|---|---|---|---|---|---|
| A1 | 缺 causal 门控 | attention_v1.py:1438 `_can_use_custom_fia` | 非 causal/双向模型被按 causal 掩码 → **静默错误输出**；full_graph 下 `weak_ref_tensors(None)` 崩溃 | `if not attn_metadata.causal: return False` | op 支持 sparse_mode=0 无 mask 路径 | ❌ |
| A2 | BATCH_INVARIANT 组合 | 同上 | 默认栈 `VLLM_BATCH_INVARIANT=1` 时 `_C_ascend` 不注册 → enabled=True 首次调用**硬崩溃** | 函数开头 `if envs_vllm.VLLM_BATCH_INVARIANT: return False` | 与 csrc 注册机制对齐 | ❌ |
| A3 | draft 图池污染 | attention_v1.py:1566 `full_graph_custom_fia` | EAGLE/MTP：draft 捕获写进主池 → 重放错配，输出损坏 | capture 分支 `if _EXTRA_CTX.is_draft_model: return False` | 镜像 baseline :996-1002 draft 池分流 | ❌ |
| A4 | layer-aware replay 错配 | attention_v1.py:858 update 分支 | gemma4 混合层：全局层重放拿 SWA 组 block_table → **静默错误输出** | `if self._use_layer_aware_fia_graph_replay: return False`（capture 侧） | 元组带 layer_name，镜像 baseline :929 解析 | ❌ |
| A5 | DevListCache 容量冻结 | torch_adpt.h:37-95 | warmup 小 batch 后 >512 并发 → **服务中每次 attention 抛异常**；eager/图共用缓存，eager 扩容会破坏已捕获图 | —（必须改代码） | 缓存按 eager/graph 拆两对 + eager 2x 扩容 + graph 容量=max_num_seqs | ❌ |
| A6 | workspace 桶无 max-sizing | attention_v1.py:1589 | SWA+全局混合层同 bucket → 后到算子拿欠尺寸 buffer 崩溃 | —（必须改代码） | 复用 `cache_graph_workspace(..., use_max_workspace=self._use_max_workspace_for_fia_graph)` | ❌ |
| A7 | 混合批数值漂移 | attention_v1.py:1471 | ChunkedPrefill 混合批：数值随 batch 组成变化 | 混合批 + phase-split 能力时 return False | 镜像 `_forward_fia_chunked_prefill_split` | ❌ |
| A8 | FD tiling 核数越界隐患 | tiling.cpp:786-790/:487 | 仅当前 SoC 20 核安全；≥27 核注册即 host 缓冲区越界写 | —（必须改代码） | FD 分支 `blockNum_ = min(blockNum_, MAX_CORE_NUM_FD)`（须在任务均分前） | ❌ |
| A9 | preload 每步同步 | torch_adpt.h:57-60 | decode 稳态每步 `aclrtSynchronizeEvent` 阻塞 host，破坏异步 run-ahead | — | eager 缓存双缓冲（前提：A5 拆分先做） | ❌（效率） |

---

## B. Kernel 既有 bug（非集成层，独立跟踪）

| # | 问题 | 状态 |
|---|---|---|
| B1 | **6b real-skip 挂死**：`sparse_lambda=-3` + stats off 在当前 kernel 上确定性挂死（新进程、无前置用例也复现）。并行会话 `repro_6b.py` 在 bisect（其"poisoning predecessor"前提不成立——已验证单用例即挂）。e2e 用例 `test_blasst_sparse_output` 已 `@pytest.mark.skip` 并注明原因 | 🔴 开放，**阻塞 real-skip 模式上线**；修复后解除 skip |
| B2 | 逐核 cycle 插桩（原 {4096} 版）已从本分支移除未重落地。**重落地前置条件**：核数上界守卫 + 槽位布局共享 constexpr（五处重复硬编码问题，检视 #8） | ⚪ 待定（是否还需要取决于 B1 排查） |

---

## C. 合入规范符合性遗留

| # | 事项 | 现状 | 剩余动作 |
|---|---|---|---|
| C1 | CI 用例看护（规范 9） | **部分完成**：e2e 精度 16 用例已注册 `test_config.yaml: 180`；config UT 进 cpu-ut | ① PR 后 CI 实测校准 180s 估算；② 图模式（full_graph capture/update）与 spec-decode 的 e2e 用例仍缺——这是 A3/A4 修复的验证载体（UT 无法覆盖图语义，PR 描述按规范 11 说明 e2e 必要性） |
| C2 | 旧测试目录清理 | 未做 | `git rm` 已提交的 7 个调试产物（`fia_ab_debug_replay_analysis_*.txt`、`precision_cases_findings.md`、`batch_*_summary.txt` ×3、`精度与E2E结果汇总.md`）+ 9 个硬编码 `/home/z00603376/` 路径脚本的去留决策（去路径化挪 `tools/` 或移除）；注意 reset 后这些文件现为 untracked，**不做 git rm、直接不加入新提交即可** |
| C3 | PR 描述（规范 10/11） | 未写 | 性能数据（加速比）、attention 状态覆盖矩阵、回退场景说明（PrefillCacheHit/SpecDecoding → baseline）、新增配置项说明、e2e 用例必要性说明 |
| C4 | commit message 规范 | `f2989fa88` 缺 Conventional Commits type/sign-off（AGENTS.md）；重落地提交时修正 | 重落地时按规范写 |
| C5 | golden 路径耦合 | e2e 测试经 `sys.path` 导入 `tests/fused_infer_attention_score/blasst_golden_tnd.py`（506 行，仓内但位置在待清理目录） | 清理 C2 时把 golden 挪到 e2e 旁（或独立 `tests/goldens/`），同步改导入路径；挪动前 e2e 已带 graceful skip 不会断 |

---

## D. 代码质量遗留

| # | 事项 | 位置 | 备注 |
|---|---|---|---|
| D1 | 死代码 `_forward_fia_slidingwindow`（29 行，零调用） | attention_v1.py:1408-1436 | 直接删 |
| D2 | 魔数收敛：`sparse_mode=3`、`2147483647`（与既有 `SWA_INT_MAX` 混用）、`-99.0`、`sparse_stats {16}` 四处字面量 | attention_v1.py / torch_adpt.h / proto.cpp / torch_binding_meta.cpp | {16} 提为 adapter 内 `constexpr kSparseStatsElems`，其余提命名常量 |
| D3 | 22 参调用三处复制 + 15 元组两处位置对齐 | attention_v1.py:864/:1515/:1648（update 分支现为 :890 附近） | **环境变量修复又给三处各加了 2 行，重复加重**——D3 的必要性上升；抽 `_custom_fia_call_kwargs()` + `CustomFIAGraphParam` dataclass |
| D4 | tiling 每调用 `OPS_LOG_I("FIA debug: ...")` 生产日志 | tiling.cpp:851 | 降级 `OPS_LOG_D` 或删 |
| D5 | get_workspace meta 占位返回 `query.size(0)` | torch_binding_meta.cpp:2128 | trace 下 mis-size；加注释/TORCH_WARN |
| D6 | `_get_fia_params` 不传 kv_cache（custom 路径 key_cache=None 时 RuntimeError，baseline 可懒初始化） | attention_v1.py:1491/:1553 | 对齐 baseline 传参 |
| D7 | 分发优先级与注释不符：paged_attention 先于 custom FIA，DecodeOnly 大多走不到 custom；注释宣称"全场景单跑" | attention_v1.py:2057-2067 与 :1462 | 调序或改注释 + config docstring 写明优先级 |
| D8 | kernel 侧 `EnableGraphApi`/交叉核事件等历史遗留（检视未列但基线普查提到的 W2 快照注释） | refactor 分支上下文 | 随 B1 排查一并梳理 |

---

## E. 环境 / 协作备忘

| # | 事项 | 说明 |
|---|---|---|
| E1 | **并行会话协调**（最紧迫） | 对方做了 `reset HEAD~` 并在 `refactor/fia-review-fixes` 有 W2 重构线。重落地前必须对齐：基底选哪个、提交怎么拆、A 类修谁来做——否则两条线会冲突 |
| E2 | 工作区未提交风险 | 见 §0；建议提交拆分：① op/csrc 算子本体 ② vllm_ascend 集成 ③ 测试（e2e+UT+yaml）④ 文档/报告不入特性提交 |
| E3 | 本机缺 `modelscope` | `tests/e2e/conftest.py` 导入它——本地 pytest 直跑会失败（用 driver 方式跑测试函数；CI 镜像无此问题）。可与维护方确认 conftest 是否该延迟导入 |
| E4 | 挂死进程强杀 → 设备 wedge | 经实测：timeout 强杀挂死的 kernel 后，设备 stream 异常数分钟（新进程变慢/输出错/507014），**自愈**无需复位；期间 npu-smi 显示 Health OK 具有迷惑性。B1 排查时避免 SIGKILL，等自愈或换 chip |
| E5 | e2e 时长估算 | 注册 180s：实测活跃用例 ~90s + NPU 初始化；B1 解除 skip 后（+4 个 golden 用例 ~4min）需上调到 ~450s |

---

## F. 已完成项（对照，勿重复）

| 项 | 内容 | 验证 |
|---|---|---|
| 环境变量并入配置 | `VLLM_FIA_HOST_SEQ_TILING`/`VLLM_FIA_FD` getenv → op attr + `CustomFIAConfig.host_seq_tiling/flash_decode`（9 文件全链路） | op 包+扩展重建；`flash_decode=False`/`host_seq_tiling=False` fallback 用例通过 |
| e2e 精度看护 | `test_custom_fia_precision.py`：dense/causal/GQA/paged/FD 形态 × fp16/bf16 + stats vs golden，16 活跃 + 2 skip | 16/16 |
| config UT | `TestCustomFIAConfig`（默认/显式/lax bool/未知键） | 2/2 |
| {16} cacheline 修复 | stats 张量 {2}→{16} 消除 cacheline 回写越界踩邻（并行会话） | e2e 通过 |
| 检视与建议文档 | 两份 repo 根文档 | — |

---

## 建议处理顺序

```
第 0 步  E1/E2：与并行会话对齐 reset 意图 → 分批提交工作区（防丢失）
第 1 步  A1-A4 + A7：五行门控 + 提交（半天，随重落地 commit）
第 2 步  A5/A6/A8：结构性修复（1-2 天）
第 3 步  B1：kernel real-skip 挂死（并行会话主攻；解除 e2e skip）
第 4 步  D1-D7 + C2/C5：质量与清理（1 天）
第 5 步  C3/C4：PR 描述与提交规范化 → 走合入流程
```
