# fused_infer_attention_score Tiling Key 与执行路径深度分析

> 本文基于 `ops-transformer` 原分支代码，对 `fused_infer_attention_score`（下文简称 FIA）算子的 host 侧 tiling 分发逻辑、各路径 tiling key 编码规则，以及 `vllm-ascend/attention/attention_v1.py` 中实际调用配置如何映射到这些路径进行系统性梳理。
> 分析范围覆盖 SplitFuse（BlasST）、PFA、IFA、V3 legacy 四条主要路径。

---

## 1. 顶层 Dispatch 流程

Host 侧入口位于 `ops-transformer/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp`：

```cpp
ge::graphStatus TilingFusedInferAttentionScore(gert::TilingContext *context) {
    if (RouteToFia(context)) {                 // ① 先判断是否走 V3 老模板
        return TilingFusedInferAttentionScoreV3(context);
    }
    ...
    bool usingIFA = IsUsingIFA(*context, inputLayoutStr, queryD, queryS);
    bool usingFAI = IsUsingFAI(*context, inputLayoutStr, queryD);
    if (usingFAI) {                            // ② SplitFuse / BlasST 路径
        TilingProcess4SplitFuse(context);
    } else if (usingIFA) {                     // ③ IFA 路径
        TilingProcess4IFA(context);
    } else {                                   // ④ PFA 路径
        TilingProcess4PFA(context, ...);
    }
}
```

Kernel 侧入口 `ops-transformer/attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score.cpp` 按 `TILING_KEY_VAR` 范围进入不同子核：

```cpp
if (TILING_KEY_VAR >= FAI_FLAG_TILING) {          // 5e18 起 → SplitFuse
    ... SplitFuse::FAInfer<...> / FAInferDecoding<...>
} else if (TILING_KEY_VAR >= PFA_FlAG_TILING) {   // 1e18 起 → PFA
    prompt_flash_attention_FIAS_arch32(...);
} else if (TILING_KEY_VAR >= FIA_FLAG_TILING) {   // 1e17 起 → V3 legacy
    fused_infer_attention(...);
} else {                                          // <1e17 → IFA
    incre_flash_attention_FIAS_arch32(...);
}
```

关键边界宏定义（`fused_infer_attention_score_tilingkey.h`）：

| 宏 | 值 | 含义 |
|---|---|---|
| `FAI_FLAG_TILING` | 5000000000000000000 | SplitFuse 起点 |
| `PFA_FlAG_TILING` | 1000000000000000000 | PFA 起点 |
| `FIA_FLAG_TILING` | 100000000000000000 | V3 / FIA legacy 起点 |

---

## 2. Tiling Key 体系总览

| 路径 | key 范围 | 生成函数 | kernel 入口 |
|---|---|---|---|
| **SplitFuse / BlasST** | `5...` / `51...` / `52...` | `FAInferTiling::GetTilingKey()` | `SplitFuse::FAInfer` / `FAInferDecoding` |
| **PFA** | `1...`（10^18 位） | `PromptFlashAttentionTiling::TilingGetTilingKeyAttentionAscendC` | `prompt_flash_attention_FIAS_arch32` |
| **IFA** | `<10^17`，常见 `1.../2.../3...`（10^16 位） | `IFATiling::GenTilingKey()` | `incre_flash_attention_FIAS_arch32` |
| **V3 legacy** | `103.../104.../105...` / `203.../204.../205...` | `FiaTilingNonQuant/NonQuantMla::GenTilingKey()` | `fused_infer_attention` |
| **Empty tensor** | 特殊注册名如 `IncreFlashAttention_13/14/27/30` | — | 对应 empty kernel |

---

## 3. SplitFuse（5...）Tiling Key 解码

生成函数：`ops-transformer/attention/fused_infer_attention_score/op_host/flash_attention_infer_tiling.h::GetTilingKey()`

```cpp
uint64_t FAInferTiling::GetTilingKey() {
    constexpr uint64_t SPLIT_FUSE_BASE_KEY      = 5000000000000000000;
    constexpr uint64_t PAGED_CACHE_KEY          = 10000000;
    constexpr uint64_t COMP_CAUSAL_MASK_KEY     = 3;
    constexpr uint64_t COMP_SWA_MASK_KEY        = 5;
    constexpr uint64_t FULL_MASK_KEY            = 6;
    constexpr uint64_t LAYOUTQ_TND_KEY          = 200000;
    constexpr uint64_t DTYPE_FP16_KEY           = 100;
    constexpr uint64_t DTYPE_BF16_KEY           = 200;
    constexpr uint64_t LSE_OUT_ONLY_KEY         = 1000;
    constexpr uint64_t INNER_LOW_PREC_KEY       = 10000;
    constexpr uint64_t LEARNABLE_SINK_KEY       = 100000000;
    constexpr uint64_t FLASH_DECODE_KEY         = 100000000000000000;  // 51...
    constexpr uint64_t DECODING_KEY             = 200000000000000000;  // 52...
    ...
}
```

### 字段解码表（固定 19 位十进制）

以 `5 0000000 00 1 020 0 1 03` 这种分段方式理解：

| 位段 | 权重/位置 | 字段 | 取值含义 |
|---|---|---|---|
| 1 | 10^18 | 路径标识 | 固定为 `5` |
| 2 | 10^17 | Decoding/FlashDecode 标识 | `0`=regular；`1`=FlashDecode（51...）；`2`=Decoding（52...） |
| 3 | 10^16 ~ 10^9 | reserved / padding | 当前多为 0 |
| 4 | 10^8 | LearnableSink | `0`=无 sink；`1`=有 sink（+100000000） |
| 5 | 10^7 ~ 10^6 | Layout | `0200000` 段：Q layout 为 TND 时 +200000 |
| 6 | 10^5 ~ 10^4 | Dtype | `100`=FP16；`200`=BF16 |
| 7 | 10^4 | InnerLowPrecise | `0`=高进度；`1`=低精度 inner（+10000） |
| 8 | 10^3 | LSE out | `0`=no LSE；`1`=LSE（+1000） |
| 9 | 10^2 ~ 10^0 | Mask type | `0`=NO_MASK；`3`=CAUSAL(COMP)；`5`=SWA；`6`=FULL |
| 10 | 10^7 | Paged cache | `0`=无 block_table；`1`=有 block_table（+10000000） |

### 典型 key 示例

| key | 含义 |
|---|---|
| `5000000000000200103` | FP16、TND、no LSE、no cache、causal mask、regular |
| `5000000000010200103` | FP16、TND、no LSE、paged cache、causal mask、regular |
| `5000000000000201103` | FP16、TND、LSE out、no cache、causal mask、regular |
| `5000000000100200103` | FP16、TND、no LSE、no cache、causal mask、+learnable sink |
| `5000000000000210100` | FP16、TND、no LSE、no cache、no mask、inner low precision |
| `5100000000010200103` | FP16、TND、no LSE、paged cache、causal mask、FlashDecode |
| `5200000000010200100` | FP16、TND、no LSE、paged cache、no mask、Decoding |
| `5000000000000200203` | BF16、TND、no LSE、no cache、causal mask、regular |

---

## 4. PFA（1...）Tiling Key 解码

生成函数：`ops-transformer/attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp::TilingGetTilingKeyAttentionAscendC()`

PFA key 采用**累加制**，不是严格十进制位权，而是多个独立偏移叠加：

| 偏移 | 触发条件 | 说明 |
|---|---|---|
| `+0/1` | 非 CVDIFF 且 SOuter/SInner/D 有 tail | tail 标记 |
| `+10` | 使用新模板 | 固定 +10 |
| `+5` | `BNSD`/`NSD` layout | layout 偏移 |
| `+100` | `inputDataType == DT_BF16` | BF16 输入 |
| `+200` | `inputDataType == DT_INT8` | INT8 输入 |
| `+1002` | `tilingMod == CVDIFF` | CV diff：+1000 并统一 +2 |
| `+10000` | `outputDataType == DT_BF16` | BF16 输出 |
| `+20000` | `outputDataType == DT_INT8` | INT8 输出 |
| `+100000` | CVDIFF 下 `BSH/SH/BSND` | CVDIFF layout 附加 |
| `+1000000` | enable matmul norm | 仅 matmul tiling 优化 |
| `+2000000` | `SPLIT_NBS_CUBE` / `SPLIT_ONEN_CUBE` | l1reuse |
| `+10000000` | `enablePA == true` | Page Attention |
| `+100000000` | `isKVHasPrefix == true` | prefix KV |
| `+400200000000` | `enableMsd == true` | MSD |
| `+8 * 10^11` | `query FP16 && key INT8 && !msd` | KV 反量化 binary flag |

### 310P 特殊 key

- `12288`：BNSD 旧模板
- `22288`：BSH/BSND 旧模板（+10000）

### 典型 key 示例

| key | 含义 |
|---|---|
| `1000000015` | new template + PA + BF16 input + BNSD layout |
| `400200001012` | MSD + CVDIFF + new template 组合 |
| `12288` / `22288` | 310P legacy 模板 |

---

## 5. IFA（<10^17）Tiling Key 解码

生成函数：`ops-transformer/attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp::GenTilingKey()`

```cpp
constexpr uint64_t IFA_TILINGKEYOFFSET          = 10000000000000000UL;  // 10^16
constexpr uint64_t IFA_PERF_MODE_TILINGKEYOFFSET = 1000000000000000UL;  // 10^15

baseOffset = modeVal * 10^16 + perfMode * 10^15;
tilingKey = baseOffset + IFA_GET_TILINGKEY(layoutVal, inputQVal, inputKvVal, outputVal, originVal,
    flagVal, antiquantModeOr0, kvLayoutVal, amlaMode, balanceMode, cvRatioVal);
```

其中 `IFA_GET_TILINGKEY` 为十进制递归和，参数依次占用更高位到更低位。

### 字段解码表

| 位权 | 字段 | 取值含义 |
|---|---|---|
| 10^16 | `modeVal` | `1`=normal；`2`=sysPrefix；`3`=ATB |
| 10^15 | `perfMode` | 0=NORMAL；1=BMM_ALL_BY_VEC；2=C1_V1；3=CUBE_VIEW_MM；4=CUBE_VIEW_MM_FULL_LOAD；5=CUBE_VIEW_MM_MLA；6=CUBE_VIEW_MM_DD |
| 10^14 | `layoutVal`（Q layout） | 0=BNSD；1=BSH/BSND；2=TND |
| 10^13 | `inputQVal` | 0=FP16；2=BF16；3=INT8 |
| 10^12 | `inputKvVal` | 0=FP16；2=BF16；3=INT8；4=INT4 |
| 10^11 | `outputVal` | 0=FP16；2=BF16；3=INT8 |
| 10^10 | `originVal` | 通常同 `inputQVal` |
| 10^9 | `flagVal` | `paVal(0/2) + splitKvVal(0/1) + antiquantModeVal(0/4)` |
| 10^8 | `antiquantMode` 或占位 | per-token/channel 分支为 0；否则填 `antiquantMode_` |
| 10^7 | `kvLayoutVal` | 0=BNSD；1=BSH/BSND；2=NZ；3=TND |
| 10^6 | `amlaMode` | 0=DISABLE；1=AMLA；2=AMLA_3BUF |
| 10^5 | `balanceMode` | 0/1 |
| 10^4 | `cvRatioVal` | 0=CV 1:2；1=CV 1:1（仅 rope+quant 场景） |

### 说明

当 `antiquantMode_ == PER_TOKEN_MODE || PER_CHANNEL_MODE` 时，反量化信息被折叠进 `flagVal`（+4），`10^8` 位置固定为 0；否则 `flagVal` 只含 PA/splitKV，`antiquantMode_` 放在 `10^8`。

### 典型 key 示例

| key | 含义 |
|---|---|
| `30000000000200000` | ATB 模式（mode=3），perf=0，BNSD，FP16/FP16/FP16，PA，无 splitKV（对应注册名 `IncreFlashAttention_30000000000200000`） |
| `100000000011020000` | normal IFA，BSH layout，PA，FP16/FP16/FP16 |

---

## 6. V3 Legacy（103/104/105...）Tiling Key 解码

生成函数：

- GQA：`attention/common/op_host/arch32/fia_tiling_nonquant.cpp::GenTilingKey()`
- MLA：`attention/common/op_host/arch32/fia_tiling_nonquant_mla.cpp::GenTilingKey()`

```cpp
constexpr uint64_t FIA_TILINGKEYOFFSET          = 100000000000000000UL;  // 10^17
constexpr uint64_t FIA_PERF_MODE_TILINGKEYOFFSET = 1000000000000000UL;   // 10^15

baseOffset = modeVal * 10^17 + perfMode * 10^15;
```

### 字段解码表

| 位权 | 字段 | 取值含义 |
|---|---|---|
| 10^17 | `modeVal` | `1`=普通；`2`=sysPrefix（sysPrefixFlag） |
| 10^15 | `perfMode`（对应 `FiaTemplateId`） | 3=HIGH_PERFORMANCE_GQA；4=GENERAL_GQA；5=HIGH_PERFORMANCE_MLA |
| 10^6 | `layoutVal`（Q layout） | GQA: 0=BNSD；1=BSH/BSND；3=TND；5=NTD<br>MLA: 0=BNSD；1=BSH/BSND；2=TND |
| 10^5 | `inputQVal` | 0=FP16；2=BF16；3=INT8；4=INT4 |
| 10^4 | `inputKvVal` | 同上 |
| 10^3 | `outputVal` | 同上 |
| 10^2 | `originVal` | 同 `inputQVal` |
| 10^1 | `flagVal` | GQA: `softmaxBrcbFlagVal(0/4) + paVal(0/2) + splitKvVal(0/1)`<br>MLA: `paVal(0/2) + splitKvVal(0/1)` |
| 10^0 | `antiquantModeVal` | 当前 GQA/MLA 非量化模板固定为 0 |
| 10^0（MLA 额外） | `cvRatioVal` | MLA 在 flagVal 后还有 `cvRatioVal` |

### 典型 key 族

| key 开头 | 含义 |
|---|---|
| `103...` | mode=1 + perf=3 → HIGH_PERFORMANCE_GQA |
| `104...` | mode=1 + perf=4 → GENERAL_GQA |
| `105...` | mode=1 + perf=5 → HIGH_PERFORMANCE_MLA |
| `203/204/205...` | sysPrefix 版本 |

### 典型 key 示例

| key | 含义 |
|---|---|
| `105000000000200000` | MLA + BNSD + FP16/FP16/FP16 + PA（对应头文件 `QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING`） |
| `105000000020322220` | MLA + BF16 + NZ KV layout + PA + flash decoding |
| `105000000000022220` | MLA + BF16 + BNSD + NonPA |

---

## 7. `attention_v1.py` 调用配置映射

文件：`/home/z00603376/blasst_model/vllm-ascend/vllm_ascend/attention/attention_v1.py`

| Python 调用位置 | 关键配置 | Host 选择 | 典型 tiling key / 说明 |
|---|---|---|---|
| `forward_fused_infer_attention` 主分支（`attention_v1.py:858`） | `input_layout="TND"`、`block_table`、`sparse_mode=3`、`antiquant_mode=-130` | `IsUsingFAI` → `TilingProcess4SplitFuse` | `5000000000010200103`（paged cache, causal mask, FP16, no LSE）或 `5000000000110200103`（+learnable sink） |
| `full_graph_fia` / graph capture 更新节点（`attention_v1.py:509/609`） | 同上，TND + block_table + sparse_mode=3 | `IsUsingFAI` → SplitFuse | 同上，key 以 `5` 开头 |
| `_forward_fia_slidingwindow`（`attention_v1.py:751`） | `input_layout="BSH"`、`queryS=1`、`block_table`、`pre_tokens=sliding_window` | `IsGqaIfa` → `IsUsingIFA` → `TilingProcess4IFA` | IFA key，例如 `1xxxxxxxxxxxxxxx`（BSH layout val=1，PA=2） |
| sinks 分支 `npu_fused_infer_attention_score_v2`（`attention_v1.py:836`） | `input_layout="TND"`、`learnable_sink`、`sparse_mode=3/4` | **v2 独立算子** | 不属于标准 `npu_fused_infer_attention_score` tiling key 体系，备注"不涉及" |
| `_forward_c8_decode`（`attention_v1.py:1205`） | `input_layout="BNSD"`、query S=1、INT8 KV + antiquant_scale/offset、`block_table`、`sparse_mode=0` | `IsGqaIfa` → `IsUsingIFA` → IFA | IFA key，例如 `1xxxxxxxxxxxxxxx`（BNSD layout val=0，inputKvVal=3，PA=2） |
| `_forward_c8_chunked_prefill` decode 部分（`attention_v1.py:1253`） | 同 `_forward_c8_decode` | `IsGqaIfa` → IFA | 同上 |
| `_forward_c8_chunked_prefill` prefill 部分（`attention_v1.py:1307`） | `input_layout="TND"`、float KV、`block_table=None`、`sparse_mode=3` | `IsUsingFAI` → SplitFuse | `5000000000000200103`（no cache, causal mask, FP16） |
| `_forward_c8_fused_infer_attention`（`attention_v1.py:1374`） | `input_layout="TND"`、float KV、可能 `block_table=None`、`sparse_mode=3` | 若 dense KV / 无 block_table：`IsUsingFAI` → SplitFuse；若仍为 INT8 paged 则先 dequant 后再走 SplitFuse | `5000000000000200103` 或类似 |

---

## 8. 全路径 scenario → tiling → kernel 入口汇总

| 输入场景 | Host 选择函数 | Tiling 路径 | Tiling Key 范围/示例 | Kernel 入口 |
|---|---|---|---|---|
| TND、有 block_table、sparse_mode=3/4/0、满足 MHA/GQA 尺寸约束 | `IsUsingFAI` | `TilingProcess4SplitFuse` | `5...` / `51...` / `52...` | `fused_infer_attention_score.cpp:53` 内 `SplitFuse::FAInfer/FAInferDecoding` |
| TND/BSH/BNSD/BSND 等 + queryS=1 或 queryD=512（GQA/MLA）、走 IFA | `IsUsingIFA`（含 `IsGqaIfa/IsMlaIfaOrMtp/IsSlidingAttention` 等） | `TilingProcess4IFA` | `<10^17`，常见 `1.../2.../3...` | `fused_infer_attention_score.cpp:660` → `incre_flash_attention_FIAS_arch32` |
| 非 SplitFuse、非 IFA 的 prefill 场景 | fallback | `TilingProcess4PFA` | `1...`（10^18 位） | `fused_infer_attention_score.cpp:644` → `prompt_flash_attention_FIAS_arch32` |
| `RouteToFia` 为 true 的 legacy 场景 | `RouteToFia` | `TilingFusedInferAttentionScoreV3` | `103.../104.../105...`<br>`203.../204.../205...` | `fused_infer_attention_score.cpp:653` → `fused_infer_attention` |
| 空 tensor / 特殊 fallback | — | empty input tiling | `IncreFlashAttention_13/14/27/30` 等 | 对应 empty kernel |

---

## 9. 关键判断函数速查

### `IsUsingFAI` 主要条件（`fused_infer_attention_score_tiling.cpp:1455`）

- `inputLayoutStr == "TND"`
- 无 `query_rope/key_rope`（非 rope split mla）
- `sparse_mode` 为 0/3/4
- MHA 或 GQA 条件满足
- Q/K/V head dim ≤ 256 且相等
- 有 PA 时 KV 为 3 维、block_size 16 对齐且 ≤ MAX_BLOCK_SIZE

### `IsUsingIFA` 主要条件（`fused_infer_attention_score_tiling.cpp:1708`）

满足以下任一子条件即进入 IFA：

- `IsGqaIfa`：TND/NTD + queryD=512；或 BSH/BNSD/BSND/NBSD 变体 + queryS=1
- `IsGqaMtp`：BSH/BNSD/BSND/TND + queryS∈[1,16] + KV 为 NZ 格式
- `IsAtbIfa`：TND/NTD + KV INT8 反量化 + PA + queryD≠512
- `IsMlaIfaOrMtp`：有 query_rope/key_rope + TND/NTD queryD=512；或 BSH 等 + queryS=1
- `IsSlidingAttention`：BSH 等 + queryD=512 + sparse_mode=4

### `RouteToFia` 主要条件（`arch32/fused_infer_attention_score_tiling_v3.cpp`）

- 非 ASCEND310P
- 非 NSD layout
- 其余 fallback 到 V3 legacy

---

## 10. 总结

- **SplitFuse（5...）**：`attention_v1.py` 主推理路径（TND + paged cache + sparse_mode=3）走的都是它，也是本次 BlasST 迁移的核心路径。
- **IFA（<10^17）**：处理 decode 场景（BNSD/BSH + S=1）、sliding window、C8 INT8 KV decode。
- **PFA（1... 在 10^18 位）**：标准 FIA 中未命中 FAI/IFA 的 prefill fallback。
- **V3（103/104/105...）**：`RouteToFia` 命中的 legacy GQA/MLA 路径，kernel 入口为 `fused_infer_attention`。
- **`npu_fused_infer_attention_score_v2`**（sinks 分支）是独立 API，不在上述 `fused_infer_attention_score` 的 tiling key 体系内。

---

## 参考文件

- `ops-transformer/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp`
- `ops-transformer/attention/fused_infer_attention_score/op_host/arch32/fused_infer_attention_score_tiling_v3.cpp`
- `ops-transformer/attention/fused_infer_attention_score/op_host/flash_attention_infer_tiling.h`
- `ops-transformer/attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score.cpp`
- `ops-transformer/attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score_tilingkey.h`
- `ops-transformer/attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp`
- `ops-transformer/attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp`
- `ops-transformer/attention/common/op_host/arch32/fia_tiling_nonquant.cpp`
- `ops-transformer/attention/common/op_host/arch32/fia_tiling_nonquant_mla.cpp`
- `ops-transformer/attention/common/op_host/fia_tiling_info.h`
- `/home/z00603376/blasst_model/vllm-ascend/vllm_ascend/attention/attention_v1.py`
