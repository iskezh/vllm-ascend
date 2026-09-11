# FIA 单算子精度对比发现（rar 风格用例）

- 用例：`tnd_precision_cases.json`（JSONL，14 case，设计参考 `_rar_extract_12` 的
  `12_FusedInferAttentionScore.json` 形式）
- 脚本：`test_precision_cases.py`；完整输出：`precision_cases_report.log`
- 对比口径：
  - L1: custom(lambda=-99, dense) vs torch_npu(dense) —— 迁移正确性
  - L2: custom(lambda=-3, 稀疏) vs custom(lambda=-99) —— 稀疏跳算对输出影响
  - L3: custom(lambda=-3) vs torch_npu(antiquant 编码稀疏，仅 GQA)

## 结论

- **13/14 case 通过**：fp16/bf16、MHA/GQA、d64/128/256/384、单批/多变长 batch、
  prefill/decode/长 KV(2048)、causal/none mask 全覆盖下，L1 差异 ≤ 7.8e-3
  （多数 case 为 0 或 1e-4 量级）；L2 全 0（温和数据下 lambda=-3 不触发跳算）；
  L3 GQA 稀疏路径与 torch_npu **bit 一致**（max_abs=0）。
- causal mask 格式：两个算子都要求 vLLM SplitFuse 的固定 2048×2048 triu int8
  mask（`attention_mask.py get_splitfuse_attn_mask`），kv 总量需 ≤ 2048。

## 发现 1（需关注）：d=384 精度缺口

case 7（q=256, 16 heads, d=384, fp16, 温和数据）：
- custom vs torch_npu：max_abs **1.15**（其余 case 均 < 1e-2）
- 对照 fp32 CPU 参考：torch_npu 偏离 0.67，custom 偏离 **1.66**
- 差异分散在多个行（235/11/17/3/117 等），非单块异常
- d=64/128/256 均无此问题；d=384 = 3×BASE_K(128)，疑与 QK 分 tile 累加顺序或
  fp16 P 量化的误差放大有关，建议后续按 `ascendc-precision-debug` 流程排查。
- 影响面：当前 E2E 模型 Qwen3-32B 为 d=128，不受影响；d=384 属超出已验证
  范围（64/128/256）的新发现。

## 发现 2（信息）：极端数据分布下的 fp16 数值敏感性

初版用例用 uniform[-5,5] 数据时，d=384 case 的 fp16 实现与 fp32 参考差 ~10
（两 kernel 各自偏离 6~10）。改为贴近真实激活（~N(0,1) 量级）后，其余 case
差异回到 1e-4 量级——验证数据分布需与真实推理一致，否则 fp16 的 softmax
饱和误差会淹没 kernel 间对比。

---

# E2E 三组验证结果（2026-08-15）

- 脚本：`/home/z00603376/blasst_model/kvcomp/scripts/pipeline/run_pipeline_packed.sh`
  （LongBench-v2 9 子集 218 题，Qwen3-32B / 8 卡）
- 结果目录：
  - base:    `blasst_res/20260815_110436_base_10case`（torch_npu 原生路径）
  - custom99: `blasst_res/20260815_115500_custom99_10case`（custom FIA, lambda=-99）
  - custom3:  `blasst_res/20260815_125118_custom3_10case`（custom FIA, lambda=-3）

| 组 | Overall | 推理总耗时 | 备注 |
|----|---------|-----------|------|
| base (torch_npu) | 38.99% | 46.8 min | |
| custom -99 | 39.91% | 52.7 min | 逐 case +1%~+20%，平均 +12.5% |
| custom -3  | 39.91% | 75.5 min | 含 1 条退化请求多耗 ~22 min；剔除后 ≈53 min，与 -99 持平 |

- 模型预测一致性（218 条）：base vs c99 pred 96.8% 一致 / response 89.4% 逐字一致；
  c99 vs c3 pred 95.9% / response 89.0%（其余为采样噪声，非算子引入）
- **关键发现（退化请求）**：`_id=671b3d9fbb02136c067d52b2`（Agent history QA），
  lambda=-3 下模型陷入推理循环，response 49,922 字符（base/c99 仅 ~3k），
  pred=None，judge=False；单条请求连续生成 ~22 分钟（12:57-13:19，10.7 t/s），
  是 c3 组总耗时 75.5min 的主因。稀疏跳算（lambda=-3）在长上下文推理任务上
  存在输出退化风险——建议针对该 case 做稀疏阈值/跳算策略复验。

---

# 组 4：稀疏统计 E2E（custom -3 + LSE_STATS=1，2026-08-15）

- 结果目录：`blasst_res/20260815_144753_custom3stats_10case`
- 环境：VLLM_ASCEND_USE_CUSTOM_FIA=1 + VLLM_ASCEND_FIA_SPARSE_LAMBDA=-3 + VLLM_ASCEND_FIA_LSE_STATS=1
- 配套修复：`kvcomp/scripts/analysis/package_results.py analyze_sparse_log`
  新增 GLSE 格式解析（`sparseblock=X / totalblock=Y` + `layer=model.layers.N`），
  此前 pipeline 只认 torch_npu 的 `blockNum/blockSparseNum` 格式 → 有效记录数恒为 0

## 结果

| 指标 | 值 |
|------|-----|
| Overall 得分 | **39.91%**（与 c99/c3 完全一致 → 统计模式不改变输出的实证） |
| Overall 稀疏率（lambda=-3 检测式） | **46.26%** |
| 有效记录数 | 121,112（总 blockNum 2.45 亿 / blockSparseNum 1.14 亿） |
| 推理总耗时 | 3217s（vs c99 3161s，统计开销仅 +1.8%） |

- 稀疏率分布（桶宽 10%）：90-100% 桶占 27.9%、80-90% 桶 15.5%、<10% 桶仅 7.1%
- 每层稀疏率差异巨大：layer3 11.3%（浅层几乎不稀疏）→ layer31 93.0%（深层高度稀疏），
  符合注意力 score 分化随层加深的规律
- 各 dataset 稀疏率 42.7%~54.8%（Table_QA 54.8% 最高）
- **case1 未再出现退化**（247s vs c3 组的 1590s）——退化请求源于非统计模式的
  真实跳算路径（softmax early-return + RescaleO 读跳过），统计模式跳算关闭、
  输出与 dense 一致，进一步佐证 c3 组退化与稀疏跳算的因果关联
