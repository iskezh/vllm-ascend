# BlasST（custom FIA）遗留事项 · 完整清单（2026-09-11 刷新版）

- **基线**：HEAD `c3297b919` + 工作区未提交改动（W2 + 全部修复）
- **验证状态**：e2e 14/14（含 real-skip 真跳）+ UT 142/142 + op 包/扩展已重建
- **结论**：**P0 正确性缺陷全部清零**。剩余 = 1 个小测试项 + 4 个架构对齐项 + 流程收尾

---

## 一、已完成（按批次，勿重复）

| 批 | 内容 |
|---|---|
| 环境变量 | `VLLM_FIA_*` getenv → op attr + `CustomFIAConfig`（后 attr 死开关已随 R1-4 精简为仅 `flash_decode`） |
| T1 门控 | `__init__` 静态能力矩阵（enabled/dtype/kv_dtype 白名单/head_size=128/sinks/SWA/BATCH_INVARIANT）+ `_can_use_custom_fia`（causal/batch≤256/key_cache/混合批/capture: full_graph+DecodeOnly+非 draft+非 layer-aware）+ `TestCustomFIAGate` 11 用例 |
| CI 看护 | 精度 e2e（14 活跃用例）注册 240s；图模式 e2e 注册 300s；config UT |
| W2（并行会话） | host-list-only：seq 进 TilingData、删 D2H/DevListCache 依赖、kernel 读 tiling buffer |
| T3 | meta 占位注释；golden 挪 e2e 目录包导入 |
| T4 | 删 DevListCache+preload_seq；FD `blockNum_` 钳位；`OPS_LOG_D` |
| R1-1/2/5 | SPDBG `#ifdef FIA_SPDBG`；q/kv 等长检查前移（消 host 堆越界读）；int8 KV 白名单 |
| A 组 | workspace max-sizing（`cache_graph_workspace`）；`sparse_stats {16}`；256 常量 UT 断言；golden 已 staged |
| B 组 | 删 `host_seq_tiling` 死开关整链；删 `devTaskMode`/`no_copy`/params 死字段；D3 调用点收敛（`CustomFIAGraphParam` dataclass + `_custom_fia_op_kwargs()`） |
| stats 删除 | **只有 real-skip**；`sparse_stats_flag` = 纯计数输出开关 |
| B1 | **real-skip 挂死已解除**（fp16/bf16 各 2 轮 + 输出/计数对齐 golden）；根因疑似混版构建缓存，见 E6 |
| D1 | `_forward_fia_slidingwindow` 死代码（并行会话删除） |

---

## 二、剩余事项

### S. 小项（随手可做，合计 < 半天）

| # | 事项 | 位置 | 修法 |
|---|---|---|---|
| S1 | **测试 golden 合并**（R2-4） | `test_custom_fia_precision.py:ref_attention` ↔ `test_attention_v1_precision.py:132 compute_sdpa_reference` | 合并进 `attention_utils.py` 共享 helper（保留 paged/varlen 前缀和支持），两份 fp32 参考防止漂移 |
| S2 | ~~forward_impl 分发优先级注释~~（D7） | ✅ **决策：保留现状**（2026-09-11）——paged_attention 保持 decode 快路径优先，custom FIA 实际覆盖 PrefillNoCache/ChunkedPrefill 与 PA 不适用的 decode 配置；`forward_impl` 分发段已加显式决策注释，调序须附带两条 decode 路径的性能对比再议 |
| S3 | **B1 根因定性实验**（可选） | — | `git stash` 暂存 stats 删除 → 仅清缓存重建 → 单跑 6b。能跑通则证实挂死=混版构建（非代码 bug），为“以后先清缓存再改代码”立下判例 |
| S4 | **golden import 失败改显式告警**（R1-3 尾巴，可选） | `test_custom_fia_precision.py` | `_golden_available=False` 时目前静默 skip；改 collection 期 `warnings.warn` 或 xfail，防止看护静默消失 |

### A. 架构对齐项（PR 阶段与 maintainer 决策）

| # | 事项 | 现状 |
|---|---|---|
| A1 | **attn_infra 双 fork**（R2-1） | BlasST 拍平版（13 文件 ~5.8k 行）与 `sparse_attention_score` 版（54 文件）已分叉（coord/rescale 差异）。短期可加互指注释；合并共享头库需 maintainer 决策 |
| A2 | **`EXEC_NPU_CMD_WS` 宏收敛**（R2-2） | 66 行拷贝自 `op_api_common.h:687`；adapter 用公共宏未 include 公共头，靠 include 顺序侥幸编译。可独立做：WS 变体挪公共头 + 显式 include（中等工作量，做完需重建回归） |
| A3 | **TilingData 固定 +4KB/launch**（R1-9） | 2×256×int64 seq 数组每次全量 memcpy/上传，decode 小 batch 纯开销。可优化为变长 payload 或 eager 走 device tensor——需与 W2 设计者（并行会话）对齐 |
| A4 | **逐核插桩重落地**（B2，可选） | 若仍需 cycle 计数定位性能：核数守卫 + 槽位共享 constexpr（检视 #8 前置条件）；否则不做 |

### P. 流程收尾（PR 化必做）

| # | 事项 | 说明 |
|---|---|---|
| P1 | **旧目录清理** `tests/fused_infer_attention_score/` | untracked。**不提交**：7 个调试产物（.txt/.md 汇总）、9 个硬编码 `/home/z00603376/` 脚本、logs/prof/`__pycache__`。保留：`TEST_DESIGN.md`（更新）、`validate_post_cleanup.py`、`repro_6b.py`（B1 观察期后可删） |
| P2 | **报告文档不入 PR** | `c3297b919` 里有 3 份检视/建议/清单 md，工作区另有 3 份（调用流程/待办/完整清单）——rebase 时全部 `git rm`/不加入（留档走本地渠道） |
| P3 | **commit message 规范化** | AGENTS.md L275：Conventional Commits（`feat(attention): ...`）+ **Signed-off-by** 必须；重写 c3297b919 时把工作区改动一并重组（建议拆分：① op/csrc ② vllm_ascend 集成 ③ 测试 ④ CI 注册） |
| P4 | **PR 描述** | 必备：实测加速比数据（**当前缺性能对照，只有精度**——需跑 perf_driver 类对照实验）；attention 状态覆盖矩阵与回退场景；`custom_fia_config` 四字段说明；e2e 必要性说明（图语义 UT 不可覆盖）；已知限制（head_size=128、batch≤256、仅 910_93、TND、sparse_mode 0/3） |
| P5 | **图模式 e2e CI 验证** | `test_custom_fia_graph.py` 本地无模型权重未跑——提交后看 CI 结果校准（Qwen3-0.6B 下载 + 300s 估算） |
| P6 | **与并行会话对齐提交** | 全部改动（W2 + 修复）都在工作区未提交；确认重组方式（单一特性提交 vs 拆分）避免双方交叉 |

---

## 三、工程备忘（已沉淀的坑）

1. **构建三级缓存不刷新**：改 kernel 后行为没变 → `rm -rf csrc/build/binary/ascend910_93/{src,bin,gen}/<op>*` 再重建（B1 “挂死”疑似就是混版 ABI）
2. **`-j640` 并发竞争**：`bgmv_expand.cpp.o` 会写坏（unknown file type）→ `MAX_JOBS=16~32`
3. **设备 wedge**：挂死进程强杀后设备异常数分钟（npu-smi 显示 OK 有迷惑性，`LazySetDevice` 报错），等自愈；避免 SIGKILL
4. **`/tmp` 满盘**：构建依赖 /tmp 临时文件，满盘时 cmake configure 即失败
5. **本机缺 modelscope**：`tests/e2e/conftest.py` import 失败 → 本地用 driver 方式跑测试函数；CI 无此问题

## 四、建议下一步

```
S1（顺手）→ A2（宏收敛，独立可验证）→ P1/P2/P3（提交重组）
→ P4 补性能数据 → 提 PR → P5 看 CI → A1/A3/A4 在评审中与 maintainer 对齐
```
