# FusedInferAttentionScore 迁移前后对比报告（Host 侧）

- 迁移前（源仓）：`ops-transformer` 分支 `origin/blasst_0720`，路径 `attention/fused_infer_attention_score/op_host/`
- 迁移后（本目录）：`vllm-ascend/csrc/fused_infer_attention_score/op_host/`
- 说明：源仓工作区 master 与 blasst_0720 在 FIA 目录差异巨大，本报告一律以 `origin/blasst_0720` 分支内容为准。

---

## 一、Host 侧整体流程对比

### 迁移前（ops-transformer blasst_0720）

```
PyTorch (torch_npu.npu_fused_infer_attention_score)
  → GE 图引擎
    ├─ InferShape: IMPL_OP_INFERSHAPE → InferShapeFusedInferAttentionScore
    │    （多 layout：BSH/BNSD/BSND/TND/NTD/NSD + rope/prefix 复杂推导，427 行）
    ├─ OpDef: OP_ADD(FusedInferAttentionScore, FusedInferAttentionScoreCompileInfo)
    │    （28 个输入、16 个属性，910b/910_93/950 三平台）
    └─ Optiling: IMPL_OP_OPTILING
         ├─ TilingInputsDataDependency: 5 个输入 {ACT_SEQ_Q, ACT_SEQ_KV,
         │    QUERY_PADDING, KV_PADDING, ACTUAL_SHARED_PREFIX_LEN}
         │    placement = {TILING_ON_HOST, TILING_ON_AICPU}
         └─ Tiling = DoOpTilingFusedInferAttentionScore
              ├─ NpuArch == DAV_3510 → TilingFusedInferAttentionScoreV2 (arch35)
              └─ 否则 → TilingFusedInferAttentionScore
                   ├─ RouteToFia → V3 (arch22 老路径)
                   ├─ IsUsingFAI → TilingProcess4SplitFuse   ← BlasST 走这里
                   │    ├─ CheckFAIAvailability（10+ 个 Check* 函数）
                   │    ├─ ConvertContextToParamsFAI（填 FAInferContext + FD/decoding 判定）
                   │    ├─ FAInferTiling.DoTiling → FAInferTilingData
                   │    ├─ workspace = 16MB + coreNum*DB*4*3*4 + splitLse + splitO
                   │    └─ SetBlockDim / SetTilingKey(GetTilingKey 内部计算)
                   ├─ IsUsingIFA → TilingProcess4IFA（MLA/MTP/GQA-IFA）
                   └─ 否则 → TilingProcess4PFA
```

### 迁移后（vllm-ascend）

```
PyTorch (torch.ops._C_ascend.npu_fused_infer_attention_score)
  → GE 图引擎
    ├─ InferShape: fused_infer_attention_score_proto.cpp
    │    （仅 TND：attention_out = query 形状；lse = (T, N, 1)，61 行）
    ├─ OpDef: OP_ADD(VllmFusedInferAttentionScore)
    │    （8 个输入、12 个属性，仅 910b/910_93）
    └─ Optiling: IMPL_OP_OPTILING
         ├─ TilingInputsDataDependency: 2 个输入 {ACT_SEQ_Q, ACT_SEQ_KV}
         │    均 TILING_ON_HOST；设备侧时用 aclrtMemcpy D2H（链 libacl_rt）
         └─ Tiling = TilingVllmFusedInferAttentionScore（单一路径，无架构/IFA/PFA 路由）
              ├─ CopySeqLengthsToHost（host/device 双分支）
              ├─ ConvertContextToFAInferContext（校验内联，不再走 Check* 函数族）
              ├─ FAInferTiling.DoTiling → FAInferTilingData（tiling 头文件与源仓一致）
              ├─ workspace = 16MB + coreNum*DB*4*3*4 + spFlagSize(新增) + splitLse + splitO
              └─ SetBlockDim（FD 时用 needCoreNum）/ SetTilingKey
```

**核心裁剪**：迁移版只保留 SplitFuse(FAI) 路径，删掉了 V2/V3 架构路由、IFA/PFA、全部 Check* 校验函数族（约 1000+ 行校验逻辑内联简化为 ~10 条硬性拒绝）。

---

## 二、文件清单对比

| 文件 | 源仓 (blasst_0720) | 迁移后 | 差异 |
|---|---|---|---|
| def.cpp | 2053 行，28 输入/16 属性 | 98 行，8 输入/12 属性 | 大量裁剪 |
| infershape / proto | infershape.cpp 427 行（多 layout） | proto.cpp 61 行（仅 TND） | 裁剪 |
| tiling.cpp | 2181 行（FAI/IFA/PFA + V2 路由） | ~460 行（仅 FAI） | 裁剪 + BlasST 增强 |
| tiling 入口头 | tiling.h + index.h + constants.h + utils.h + info_parser（1630 行）+ compile_info.h + register.cpp | 仅 tiling.h 58 行 | 5 个文件合并 |
| flash_attention_infer_tiling.h | 692 行 | 688 行 | **实质一致**（仅 include/注释差异） |
| arch22/arch35/arch38、checkers/、op_api/aclnn* | 存在 | 全部未迁移 | 裁剪 |

---

## 三、逐函数对比

### 1. OpDef（def.cpp）

| 项 | 源仓 | 迁移后 |
|---|---|---|
| 类名 | `FusedInferAttentionScore` | `VllmFusedInferAttentionScore` |
| 输入数 | 28（含 dequant/quant/antiquant/rope/prefix/learnable_sink/padding/start_idx 等） | 8（q/k/v/pse/mask/act_q/act_kv/blocktable） |
| block table 输入名 | `block_table`（index 14） | `blocktable`（index 7） |
| softmax_lse 输出 | **REQUIRED** | OPTIONAL |
| `input_layout` 默认值 | `"BSH"` | `"TND"` |
| `scale` 属性 | OPTIONAL | **REQUIRED** |
| `inner_precise` 默认值 | 1 | 0 |
| sparse_lambda | **无独立属性**，复用 antiquant_mode<0 编码：`sparseLamda=(antiquantMode+100)/10` | **新增 `sparse_lambda` Float 属性**（index 10，默认 -99.0） |
| 平台配置 | 910b / 910_93 / 950 | 仅 910b / 910_93 |
| OP_ADD | 带 CompileInfo 模板参数 | 不带 |

### 2. InferShape

- 源仓 `InferShapeFusedInferAttentionScore`：按 layout 分发 GetQueryBSND/GetQueryTND/InferAttentionOutShape/InferLseOutShape，支持 out_dtype 转换映射表。
- 迁移后 `InferShapeVllmFusedInferAttentionScore`：TND 单分支，`attention_out = query`，`lse = (T, N, 1)`；InferDataType 固定 out0=query dtype、out1=FLOAT。**语义对 TND 场景等价**。

### 3. Tiling 入口

| 项 | 源仓 `DoOpTilingFusedInferAttentionScore` | 迁移后 `TilingVllmFusedInferAttentionScore` |
|---|---|---|
| 架构路由 | DAV_3510→V2，其余→V1；V1 内再路由 FAI/IFA/PFA | 无路由，直接 FAI 流程 |
| ScheduleMode | `SetScheduleMode(1)`（batch mode，SplitFuse 内） | 同样 `SetScheduleMode(1)` |
| seq 长度获取 | `GetData<int64_t>()` 直接读（依赖 TILING_ON_HOST/AICPU 框架搬运） | 显式 `CopySeqLengthsToHost`，device 侧 `aclrtMemcpy` D2H |
| BlockDim | FD 时 needCoreNum==0 用 coreNum | 相同逻辑 |
| TilingKey | `faTiling.GetTilingKey()`（头文件内计算） | 相同（头文件一致） |

### 4. 上下文转换：`ConvertContextToParamsFAI` vs `ConvertContextToFAInferContext`

| 逻辑点 | 源仓 | 迁移后 | 评估 |
|---|---|---|---|
| sparseLamda | `(antiquantMode+100)/10`（antiquant<0 时） | 直接读 `sparse_lambda` 属性 | 接口显式化，语义等价 |
| maskType | pse_shift 三分支（NO/SPEC-SWA/FULL_MASK + pseQ/pseKv） | `sparseMode==3 ? MASK_SPEC : NO_MASK`；**硬拒绝 pse_shift** | 裁剪，MASK_SPEC 映射一致 |
| FD 判定 KV 长度 | minKVSeqlen（batch 内最小值） | minKvSeqlen（已与源仓拉齐） | ✅ 一致 |
| FD 判定 embeddingSize | `embeddingSize <= 128` | `embeddingSize <= 128`（已补齐） | ✅ 一致 |
| FD 判定 mask | 排除 FULL_MASK/SWA_MASK | 显式排除 FULL_MASK/SWA_MASK（已补齐） | ✅ 一致 |
| decodingFlag 判定 | 额外要求 `batch>=aicoreNum && blockSize==128 && N/Nkv<=128 && innerPrecise==0` | 仅 `paged && maxQ==1 && minQ==1 && NO_MASK && !lse` | ⚠️ **放宽**：blockSize≠128 或小 batch 也会走 decoding 内核 |
| numKvHeads==0 | 直接用 attr 值 | `==0 时取 numHeads`（MHA 兜底） | 迁移版更健壮 |
| 校验体系 | CheckFAIAvailability 等 10+ 函数（软/硬检查） | 内联硬拒绝：dtype FP16/BF16、3D TND、sparse_mode∈{0,3}、antiquant==0、inner_precise==0、无 pse、act_seq 必须 INT64 | 裁剪为白名单式 |
| learnableSinkFlag | 读 LEARNABLE_SINK 输入 | 固定 false | 裁剪 |

### 5. Flash-Decode（FD）判定条件（拉齐后）

两侧完全一致：

```
pagedCacheFlag && !lseFlag
&& maskType ∉ {FULL_MASK, SWA_MASK}
&& !learnableSinkFlag && innerPrecise != 1
&& embeddingSize <= 128
&& maxQSeqlen * (numHeads / kvHeads) <= 128
&& maxQSeqlen <= 16
&& minKvSeqlen >= 1024          // batch 内最小值
&& minQSeqlen > 0
&& ( isLongSeq || isShortSeq )
   isLongSeq  = numTasks <= 0.8*aicoreNum && minKvSeqlen >= aicoreNum*512
   isShortSeq = numTasks <= 0.4*aicoreNum && minKvSeqlen >= 1024
   numTasks   = batch * kvHeads
```

> 变更记录：迁移初版曾使用 maxKvSeqlen 且缺 embeddingSize/mask 约束（属放宽），
> 已按源仓 `ConvertContextToParamsFAI` 逐条拉齐，并通过
> `SOC_VERSION=ascend910_9391 python setup.py build_ext` 编译验证（exit=0）。

### 6. Tiling Key 计算

- 两侧算法**完全一致**（`flash_attention_infer_tiling.h` 实质逐行相同）：
  `BASE(5e18) + PAGED(1e7) + MASK(3/5/6) + TND(2e5) + DTYPE(1e2/2e2) + LSE(1e3) + LOW_PREC(1e4) + SINK(1e8) + FD(1e17)/DECODING(2e17)`。
- 迁移后 tiling.cpp 里有一段**重复的死代码**（局部计算 `key` 仅用于 OPS_LOG_I 调试日志），不影响 `SetTilingKey(faTiling.GetTilingKey())` 的实际结果。

### 7. Workspace 公式

- 源仓：`16MB + coreNum*WORKSPACE_BLOCK_SIZE_DB*4*3*4 + splitLseTotalSize + splitOTotalSize`
- 迁移后：额外加 `coreNum * 2 * 32 * (PRELANCH_NUM + 1)`（**spFlagSize**，BlasST 稀疏标志 buffer）——BlasST 功能必需的新增项，正确。

### 8. REGISTER_TILING_DATA_CLASS 覆盖

- 源仓注册 40+ key（含 FD `51xxx`、SINK、LOW_PREC、FULL_MASK 6 等全组合）。
- 迁移后注册 **23 个 key**：非 paged 16 个 + paged 8 个 + decoding 2 个（`5200000000010200100/200`）。
- ⚠️ **缺口**：FD key（`5100000000010200100/103/200/203`）在 kernel 侧已加 `TILING_KEY_IS` + dispatch 宏，tiling 也会经 `flashDecodeFlag` 算出该 key，但 **host 侧未注册对应 TILING_DATA_CLASS**。FD 场景在 GE 图模式下可能找不到 tiling data class；aclnn 直发模式不受影响，但属迁移不完整点。

---

## 四、结论

1. **功能等价的核心路径**：TND + FP16/BF16 + sparse_mode 0/3 + BlasST sparse_lambda 场景下，tiling 数值计算（FAInferTiling 头文件）与源仓逐行一致，workspace 正确补齐了 spFlagSize，FD 判定条件已完全拉齐。
2. **有意裁剪**：多 layout、IFA/PFA、量化/rope/prefix/sink、arch35/950、全部 Check* 函数族、aclnn 层均未迁移，属设计内的最小化迁移。
3. **一处行为放宽需知悉**：decodingFlag 去掉了 `batch>=aicoreNum / blockSize==128 / group<=128 / innerPrecise==0` 约束——非常规 blockSize 或小 batch 场景路由结果可能与源仓不同。
4. **一个待补缺口**：FD tiling key 的 host 侧 `REGISTER_TILING_DATA_CLASS` 未注册（kernel 分发已就绪，host 注册缺 4 行）。
