# BlasST 待办工作清单（对照修改版）

- **日期**：2026-09-08（基于 HEAD=`c3297b919` + 工作区 W2 未提交改动）
- **用法**：按组推进，每组做完跑一次"组验证"。行号为当前工作区行号；被并行会话占用的文件以**函数锚点**为准。
- **详细缺陷背景**见《BlasST_代码检视报告》《BlasST_修改建议_合入前修复》。

## 0. 冲突地图（先看，避免和并行会话对撞）

| 文件 | 状态 | 结论 |
|---|---|---|
| `csrc/.../op_host/fused_infer_attention_score_tiling.{cpp,h}` | 🔴 **W2 改造中**（未提交，host-list 进 TilingData、删 `CopySeqLengthsToHost`/device seq 输入） | A8/D4 等 W2 落地后再改 |
| `csrc/.../fused_infer_attention_score_torch_adpt.h` | 🔴 W2 改造中（24 行未提交） | A5/A9 演变为"直接删 DevListCache"，等 W2 |
| `csrc/.../op_kernel/fused_infer_attention_score_kernel.h` | 🔴 W2 有 4 行改动 | B1/B2 并行会话主攻 |
| `vllm_ascend/attention/attention_v1.py` | 🟡 W2 只改了 3 处注释区（:834-870 update 分支、:1502-1512、:1554-1564） | **`_can_use_custom_fia`、死代码、forward_impl 区域 W2 未碰，可安全改**；改完与 W2 rebase 协调 |
| 其余（bindings/meta/config/tests/yaml） | 🟢 无人占用 | 随时可改 |

---

## T1 ~~现在就改：`_can_use_custom_fia` 五个门控（A1-A4 + A7）~~ ✅ 已完成（2026-09-08）

**实现**（已落在工作区，含比原计划更多的条件）：
- **静态能力矩阵**：`__init__` 里一次算完 `_custom_fia_supported`（enabled / model dtype∈{fp16,bf16} / kv_cache 无量化 / head_size==128 / 无 sinks / 无 sliding_window / 非 BATCH_INVARIANT），容错 ascend config 未初始化的 UT 构造场景
- **per-call 门控**：`_can_use_custom_fia` 重写——causal / batch≤`CUSTOM_FIA_MAX_HOST_SEQ`(256) / cache 状态须已绑定 key_cache / 混合批+phase-split 回退 / capture 须 full_graph+DecodeOnly+非 draft+非 layer-aware / attn_state 白名单
- **UT 锁定**：`tests/ut/attention/test_attention_v1.py` 新增 `TestCustomFIAGate` 10 用例（每条门控正反向），全文件 43/43 通过

**位置**：`vllm_ascend/attention/attention_v1.py:1438` `def _can_use_custom_fia`（W2 未触碰此函数）

在 `if not fia_config.enabled: return False` 之后、`if _EXTRA_CTX.capturing:` 之前插入（含 2026-09-08 入参支持矩阵分析新增的 batch/head_size 两项）：

```python
        # ── op 侧硬边界的 Python 预检（缺省即回退 baseline）──
        # ① head_dim：kernel GEMM tile 按 128 设计，全部测试仅覆盖 D=128，
        #    op 无显式校验（未声明行为），非 128 不放行
        if self.head_size != 128:
            return False
        # ② batch 上限：tiling host-list 内置数组 FIA_MAX_HOST_SEQ_LIST=256
        #    （tiling.cpp OPS_ERR_IF），max_num_seqs 可配 >256，超限回退
        if len(attn_metadata.actual_seq_lengths_q) > 256:
            return False
        # ③ batch-invariant 模式下 _C_ascend 自定义算子不注册（csrc 构建开关），
        #    且本路径未实现分相处理（见 baseline 的 phase split），回退 baseline。
        if envs_vllm.VLLM_BATCH_INVARIANT:
            return False
        # ④ custom op 仅实现 sparse_mode=3（causal+mask）与等长 dense；tiling 对
        #    sparse_mode==3 恒映射 MASK_SPEC 且不校验 mask 存在性——非 causal/
        #    交叉注意力（ENCODER/ENCODER_DECODER）会拿到 causal 掩码静默算错
        if not attn_metadata.causal:
            return False
        # ⑤ 混合 decode+prefill 的 ChunkedPrefill 批：baseline 在 batch-invariant
        #    或 CHUNKED_PREFILL_PHASE_SPLIT 下分相处理保证批不变性，custom 未实现。
        if (attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill
                and attn_metadata.num_decodes > 0
                and attn_metadata.num_prefills > 0
                and (envs_vllm.VLLM_BATCH_INVARIANT
                     or get_current_hardware_profile().supports(
                         HardwareCapability.CHUNKED_PREFILL_PHASE_SPLIT))):
            return False
```

在 capture 分支内（`if not (fia_config.full_graph and ...): return False` 之后）追加：

```python
            if _EXTRA_CTX.is_draft_model:
                # draft 图池分流未实现（baseline full_graph_fia 有，见其
                # get_draft_graph_params 选择逻辑），draft 仍走 baseline
                return False
            if self._use_layer_aware_fia_graph_replay:
                # update 分支未实现 layer_name 感知解析（baseline 有），
                # 混合注意力组模型上图重放会跨层错配 metadata
                return False
```

同时**修正函数 docstring 和 :1462 附近的注释**：删掉"全场景单跑 custom：……（含 decode+prefill chunk 混合 batch）"的表述，改为准确描述（DecodeOnly/PrefillNoCache/ChunkedPrefill 且 causal、非 draft、非 layer-aware 时接管；DecodeOnly 还受 forward_impl 中 paged_attention 优先级影响——见 T2-D7）。

**导入确认**（文件头部已有则跳过）：`envs_vllm`（:24 已有）、`get_current_hardware_profile`/`HardwareCapability`（:66 已有）、`_use_layer_aware_fia_graph_replay`（:503 已初始化为 self 属性）。

**组验证**：`python3 -m py_compile vllm_ascend/attention/attention_v1.py`；回归 e2e（见 §V）。

## T2 现在就改：attention_v1.py 内的独立小项

| # | 位置 | 改法 |
|---|---|---|
| D1 死代码 | `_forward_fia_slidingwindow`（:1408-1436，29 行零调用） | 整个方法删除 |
| D6 kv_cache | `:1491` 与 `:1564` 两处 `self._get_fia_params(key, value, attn_metadata)` | 补第 4 参 `kv_cache`（函数签名 `:1348` 已支持，baseline 传法见 :1709）；`forward_custom_fused_infer_attention`/`full_graph_custom_fia` 需相应接收并透传 `kv_cache`（调用点在 `forward_impl` ~:2060 区域） |
| D7 分发优先级 | `forward_impl`（~:2057-2067）与 `_can_use_custom_fia` docstring | 二选一：**推荐改注释**——说明"DecodeOnly 且 `using_paged_attention=False` 时才接管 decode"；若要 custom 真正接管 decode，把 `_can_use_custom_fia` 分支挪到 paged_attention 之前（影响面=开启 custom_fia 的用户，当前仅自测，可接受） |
| D2a {16} 一致性 | `full_graph_custom_fia` 内 `:1581` `sparse_stats = torch.empty(2, ...)` | 改 `torch.empty(16, ...)` 并加注释"与 adapter/proto 的 kSparseStatsElems 对齐（cacheline 回写安全下限）；图内 sparse_stats_flag=False 不写入"。graph 路径 flag=False 时不写所以 {2} 不出错，但不一致留坑 |

## T3 独立文件，随时可改

| # | 文件:位置 | 改法 |
|---|---|---|
| D5 meta 占位 | `csrc/torch_binding_meta.cpp` `npu_fused_infer_attention_score_get_workspace_meta`（~:2135） | 在 `return query.size(0);` 前加注释 + 一次性告警：`TORCH_WARN(false, "get_workspace meta returns a placeholder; real size must come from the real op");`（或仅注释，若告警会污染 CI 日志） |
| C5 golden 挪家 | `tests/fused_infer_attention_score/blasst_golden_tnd.py` → `tests/e2e/pull_request/one_card/` | `git mv` 后同步改 `test_custom_fia_precision.py` 顶部 `_GOLDEN_DIR` 导入（挪家后可删掉 sys.path 逻辑直接 import）；旧目录的其他脚本见 T5-C2 |
| C1b e2e 补充 | `tests/e2e/pull_request/one_card/test_custom_fia_graph.py`（新建，待 T1 落地后） | 图模式最小用例：`custom_fia_config={enabled,full_graph}` 下 capture→update→replay 输出与 eager 一致（这是 A3/A4 门控的验证载体；spec-decode 场景确认门控生效即可暂不加正例）。完成后注册 test_config.yaml |

## T4 等 W2 落地后改（当前文件被占用）

| # | 内容 | 说明 |
|---|---|---|
| A5/A9 | **删除 `DevListCache` 整套**（`torch_adpt.h:30-104`：struct/upload_seq_lengths/seq_len_caches/seq_dev_view/preload_seq） | W2 的 host-list-in-TilingData 方案取代了 pinned 缓存设计——**不要按旧方案改"可增长缓存"**。W2 合入后：① 确认 adapter 不再引用 caches；② 删 `preload_seq` 的 schema/impl/meta（`torch_binding.cpp`、`torch_binding_meta.cpp`）与 attention_v1.py 的 preload_seq 调用注释；③ 删除后跑 e2e |
| A8 FD 核数钳位 | `tiling.cpp` FD 选中分支（当前 ~:770 `faInfo.flashDecodeFlag = true` 处）之后、任务均分（~:490 附近 `perCoreTaskNum = (totalTaskNum + blockNum_ - 1) / blockNum_`）之前 | `blockNum_ = std::min(blockNum_, static_cast<uint32_t>(MAX_CORE_NUM_FD));`（必须钳在均分之前，否则任务截断丢失；仅 FD 分支） |
| D3 调用点收敛 | 三处 24 参调用（update 分支 ~:869、eager ~:1515、capture ~:1660） | 抽 `_custom_fia_call_kwargs()`（环境变量修复又加了 2 参，重复已加重）；15 元组改 `CustomFIAGraphParam` dataclass。注意 W2 正在动 update 分支，等它落地 |
| D4 tiling 日志 | `tiling.cpp:822` `OPS_LOG_I(... "FIA debug: ...")` | 改 `OPS_LOG_D` 或删除 |
| B2 插桩重落地 | kernel.h | 若仍需要逐核计数：加核数守卫 + 槽位布局共享 constexpr（详见检视报告 #8）；否则不做 |

## R1 第二轮检视发现（2026-09-08，范围=c3297b919 + W2 未提交改造，9 项）

| # | 发现 | 状态 |
|---|---|---|
| R1-1 | **SPDBG 调试 printf 常开**（`epilogue_online_softmax.hpp:648`、`kernel.h:425`，`#if 1`）——每次稀疏判定/每核收尾都 device printf，串行化流水线且 graph capture 不安全 | ✅ 已修：改为 `#ifdef FIA_SPDBG`（保留调试能力，默认关闭） |
| R1-2 | **tiling 堆越界读**：batch=q 列表长，q/kv 等长检查在 `ConvertContextToFAInferContext`（会索引 kv[b]）之后才跑——直接调用 op 传不等长列表即越界 | ✅ 已修：等长检查移到 :694 batch 推导紧邻处，删除晚检查 |
| R1-3 | **golden 文件 untracked**：`blasst_golden_tnd.py` 未 git add 时 import 失败 → `_golden_available=False` → 精度看护静默消失 | ⏳ 提交时 `git add tests/e2e/pull_request/one_card/blasst_golden_tnd.py`；建议同时把 graceful skip 改为显式 xfail 或加 collection 期可见告警 |
| R1-4 | **`host_seq_tiling` 死开关**：W2 删了 D2H 回退后 attr 无人读，但 def/aclnn/adapter/binding/config 全链路保留，docstring 还声称"关闭回退 D2H" | ⏳ 建议删除整条链路（def attr 15 + aclnn/adapter 签名 + binding schema + `CustomFIAConfig.host_seq_tiling` + docstring）；删前需 op 包+扩展重建 |
| R1-5 | **int8 KV cache 穿过量化门禁**（vllm 0.26 `get_kv_quant_mode("int8")==NONE`）——C8 int8 模型会把 int8 cache 喂给只收 FP16/BF16 的 op，运行期硬失败 | ✅ 已修：静态矩阵改显式白名单 `kv_cache_dtype in ("auto","float16","bfloat16")`；UT 补 `kv_dtype="int8"` 用例，43/43 |
| R1-6 | **256 常量两处手工同步**（Python `CUSTOM_FIA_MAX_HOST_SEQ` ↔ C++ `FIA_MAX_HOST_SEQ_LIST`） | ⏳ 已互相注释引用；可选加 UT 硬编码断言双端值相等 |
| R1-7 | **重构死代码**：`FAIKernelParams.actualQseqlen/actualKvseqlen`（恒 nullptr 仍被填充）、`fia_exec_common` 的 `no_copy`（已 (void)）、`devTaskMode` 分支不可达 | ⏳ 待清理（删除 params 字段需同步 kernel 入口签名） |
| R1-8 | commit message 不符合 AGENTS.md（无 sign-off、非 Conventional Commits） | ⏳ 即 T5-C4b |
| R1-9 | TilingData 固定 +4KB（2×256×int64）每次 launch 全量 memcpy/上传，decode 小 batch 是纯开销 | ⏳ W2 设计权衡，eval 后决定（可优化为变长 payload 或 eager 走 device tensor） |

**检视确认无问题的点**：FD 钳位与 `SetBlockDim` 一致不丢任务；aclnn host attr 四条 adapter 路径透传正确；无 `preload_seq`/`_forward_fia_slidingwindow` 残留调用；kernel 无对已置空 `params.actualQseqlen` 的解引用；`_EXTRA_CTX` 属性默认值安全；新 UT mock 目标与构造签名全部匹配。

## R2 复用角度检视发现（2026-09-09，迟到的 finder 子代理，5 项）

| # | 发现 | 处置 |
|---|---|---|
| R2-1 | **attn_infra 双 fork**：`fused_infer_attention_score/op_kernel/attn_infra/`（13 文件 ~5.8k 行，已拍平）与仓内既有 `sparse_attention_score/op_kernel/attn_infra`（54 文件）已分叉（coord.hpp 的 MakeCoord、rescale epilogue 差异）——一边修了数值 bug 另一边不会带上 | ⏳ 合并为共享头库是大工程，建议 PR 阶段与 maintainer 对齐（短期：在双 fork 各加互指注释"修改须同步"） |
| R2-2 | **`EXEC_NPU_CMD_WS` 是 `EXEC_NPU_CMD` 的 66 行拷贝**（仅 workspace 供给方式不同）；且 torch_adpt.h 未 include `op_api_common.h`，靠 torch_binding.cpp 的 include 顺序侥幸编译 | ⏳ 建议把 WS 变体挪进 `op_api_common.h` 并在 adapter 显式 include |
| R2-3 | 256 常量双语言手工同步 | = R1-6（已登记） |
| R2-4 | e2e 的 `ref_attention` 与同目录 `test_attention_v1_precision.py:132` 的 `compute_sdpa_reference` 是同一份 fp32 golden 的两个实现（注意：本用例支持 paged/varlen 前缀和，合并时需扩参数） | ⏳ 合并进 `attention_utils.py` 共享 helper |
| R2-5 | `blasst_golden_tnd.py` 与旧目录 `FIA_splitFuse_golden2_sink_swa_inf.py`（835 行）是跳块语义的第二个仿真模型——kernel 判定规则一变需要三处（kernel/两 golden）同步 | ⏳ 旧目录按 T5-C2 清理后自然只剩一份；注意清理时不要误删成为唯一 golden 的那套 |

## T5 Kernel 与流程（并行会话主攻 / 你收尾）

| # | 内容 | 说明 |
|---|---|---|
| B1 | **6b real-skip 挂死**（`sparse_lambda=-3`、stats off，单用例即挂——bisect 脚本的"前置投毒"前提不成立） | 并行会话 `repro_6b.py` 在追。修复后：解除 `test_custom_fia_precision.py` 的 `@pytest.mark.skip`（grep `known kernel hang` 定位），并把 test_config.yaml 的 180s 上调至 ~450s（+4 个 golden 用例各 ~45s） |
| C2 | 清理 `tests/fused_infer_attention_score/`（现为 untracked） | **不要提交**以下内容：7 个调试产物（`fia_ab_debug_replay_analysis_*.txt`、`precision_cases_findings.md`、`batch_*_summary.txt` ×3、`精度与E2E结果汇总.md`）、9 个硬编码 `/home/z00603376/` 路径脚本（`perf_driver.py`、`memcheck_driver.py` 等）、logs/prof/`__pycache__`。保留提交：`blasst_golden_tnd.py`（若做 T3-C5 挪家则从旧目录删）、`TEST_DESIGN.md`（更新后）、`validate_post_cleanup.py`、`repro_6b.py`（B1 修复期保留） |
| C4 | **从提交中移除三份报告文档** | `c3297b919` 把《代码检视报告》《修改建议》《遗留问题清单》提交进了仓——合入前 `git rm` 这三个 md（工作留档走本地/评审渠道，不入特性 PR） |
| C3 | PR 描述 | 按 12 条规范写：目标与实测加速比、attention 状态覆盖矩阵与回退场景（PrefillCacheHit/SpecDecoding→baseline；非 causal/draft/layer-aware/mixed-batch→baseline）、`custom_fia_config` 三+二个字段说明（含 host_seq_tiling/flash_decode 取代 VLLM_FIA_* env）、e2e 用例必要性说明（图语义 UT 无法覆盖）、已知限制（real-skip 挂死若未修需声明 sparse_lambda 仅支持 -99/统计模式） |
| C4b | commit message | 重写 `c3297b919`（rebase 时）为 Conventional Commits 格式 + sign-off（AGENTS.md 要求） |

## V. 验证命令（每组做完跑）

```bash
# 1. 语法/静态
python3 -m py_compile vllm_ascend/attention/attention_v1.py

# 2. CPU config UT（改 ascend_config 后）
python3 -m pytest tests/ut/test_ascend_config.py -k CustomFIA -x -q

# 3. NPU e2e 精度回归（本机缺 modelscope，conftest 装不上 → 用 driver 方式；
#    real-skip 用例已 skip，不会触发 6b 挂死）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 - <<'EOF'
import importlib.util, torch
spec = importlib.util.spec_from_file_location(
    "t", "tests/e2e/pull_request/one_card/test_custom_fia_precision.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for name, fn in [("dense_no_mask", m.test_dense_no_mask),
                 ("dense_causal", m.test_dense_causal),
                 ("gqa", m.test_gqa),
                 ("paged_fd", m.test_paged_decode_flash_decode_shape),
                 ("paged_fd_off", m.test_paged_decode_flash_decode_disabled),
                 ("host_seq_off", m.test_paged_decode_host_seq_tiling_off),
                 ("paged_reg", m.test_paged_decode_regular),
                 ("blasst_stats", m.test_blasst_stats)]:
    for dt in (torch.float16, torch.bfloat16):
        fn(dt); print("PASS", name)
print("ALL PASS")
EOF

# 4. 改 csrc（tiling/adapter/bindings）后必须重建并重复 3：
bash csrc/build_aclnn.sh $PWD ascend910_9382          # op 包（~10min）
python3 setup.py build_ext --inplace                  # torch 扩展
```

**操作警示**：NPU 上挂死进程被强杀后设备会 wedge 数分钟（npu-smi 仍显示 OK，有迷惑性），期间新进程变慢/输出错/507014，**等自愈**再下结论；避免 SIGKILL。

**构建陷阱（2026-09-10 实测）**：改 kernel 源码后"构建了但行为没变"——构建系统把源码拷到 `csrc/build/binary/ascend910_93/src/` 后增量构建**不刷新**（`*_src_copy.done` 戳记），kernel 二进制还有独立缓存（`bin/`、`gen/*.done`）。清除后重建：`rm -rf csrc/build/binary/ascend910_93/{src,bin}/<op>* csrc/build/binary/ascend910_93/gen/<op>*`。验证标志：回归时 debug printf 仍在 = 跑了陈旧二进制。

## T4 完成情况（2026-09-10 ✅）

T3（D5/C5）、T4a（删 DevListCache+preload_seq）、T4b（FD 钳位+日志降级）、R1-1/2/5（SPDBG 门控、等长检查前移、int8 白名单）已全部落地；op 包 + torch 扩展重建完成；**e2e 16/16 + UT 141 全绿**。剩余：D3（调用点收敛）、C1b（图模式 e2e）、R1-3/4/6/7/9、R2 五项、T5 流程收尾。

## 推进顺序

```
T1（五个门控，~0.5 天）→ T2（四个小项，~0.5 天）→ T3（独立文件，~0.5 天）
                          ↓（与并行会话 W2 落地并行）
T4（W2 落地后：删 DevListCache / FD 钳位 / 调用点收敛，~1 天）
T5（B1 由并行会话主攻；C2/C4/C4b 在 PR 化时收尾）
```
