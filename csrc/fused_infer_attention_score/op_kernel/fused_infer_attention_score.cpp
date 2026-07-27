/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
* \file fused_infer_attention_score.cpp
* \brief Kernel entry dispatch for VllmFusedInferAttentionScore.
*/

#include "kernel_operator.h"
#include "adv_api/matmul/matmul_intf.h"
#include "fused_infer_attention_score_tilingkey.h"
#include "flash_attention_interface.cpp"

using namespace AscendC;

#define DISPATCH_FA_INFER(KEY, DTYPE_Q, DTYPE_KV, MASK_ENUM) \
    if (TILING_KEY_VAR == KEY) { \
        SplitFuse::FAInfer<DTYPE_Q, DTYPE_KV, float, false, false, \
                           MASK_ENUM, FaiKernel::inputLayout::TND>( \
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse, \
            actualSeqLengths, actualSeqLengthsKV, user, tiling, sink); \
    }

#define DISPATCH_FA_INFER_LSE(KEY, DTYPE_Q, DTYPE_KV, MASK_ENUM) \
    if (TILING_KEY_VAR == KEY) { \
        SplitFuse::FAInfer<DTYPE_Q, DTYPE_KV, float, false, false, \
                           MASK_ENUM, FaiKernel::inputLayout::TND, \
                           NpuArch::Epilogue::LseMode::OUT_ONLY>( \
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse, \
            actualSeqLengths, actualSeqLengthsKV, user, tiling, sink); \
    }

#define DISPATCH_FA_INFER_DECODE(KEY, DTYPE_Q, DTYPE_KV) \
    if (TILING_KEY_VAR == KEY) { \
        SplitFuse::FAInferDecoding<DTYPE_Q, DTYPE_KV, float, true, \
                                   FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>( \
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse, \
            actualSeqLengths, actualSeqLengthsKV, user, tiling, sink); \
    }

#define DISPATCH_FA_INFER_PAGED(KEY, DTYPE_Q, DTYPE_KV, MASK_ENUM) \
    if (TILING_KEY_VAR == KEY) { \
        SplitFuse::FAInfer<DTYPE_Q, DTYPE_KV, float, true, false, \
                           MASK_ENUM, FaiKernel::inputLayout::TND>( \
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse, \
            actualSeqLengths, actualSeqLengthsKV, user, tiling, sink); \
    }

#define DISPATCH_FA_INFER_PAGED_LSE(KEY, DTYPE_Q, DTYPE_KV, MASK_ENUM) \
    if (TILING_KEY_VAR == KEY) { \
        SplitFuse::FAInfer<DTYPE_Q, DTYPE_KV, float, true, false, \
                           MASK_ENUM, FaiKernel::inputLayout::TND, \
                           NpuArch::Epilogue::LseMode::OUT_ONLY>( \
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse, \
            actualSeqLengths, actualSeqLengthsKV, user, tiling, sink); \
    }

extern "C" __global__ __aicore__ void vllm_fused_infer_attention_score(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *pse_shift, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKV,
    __gm__ uint8_t *blocktable, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    __gm__ uint8_t *sink = nullptr;

    // Tiling-key discovery for the compile framework. These statements expand to
    // (g_tilingKey == (<key>)) in the preprocessed source so that the framework
    // registers every supported key and generates the FAInferTilingData struct.
    TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_LOW_PREC_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_LOW_PREC_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_NOMASK_LOW_PREC_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_NOMASK_LOW_PREC_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_DECODING_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_DECODING_TILING);

    // Dispatch using runtime-evaluated tiling key (the if-body is compiled per key above).
    DISPATCH_FA_INFER(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING,
                      half, half, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER_LSE(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING,
                          half, half, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING,
                      half, half, FaiKernel::MaskType::MASK_CAUSAL);
    DISPATCH_FA_INFER_LSE(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING,
                          half, half, FaiKernel::MaskType::MASK_CAUSAL);
    DISPATCH_FA_INFER(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING,
                      half, half, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER_LSE(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING,
                          half, half, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING,
                      half, half, FaiKernel::MaskType::MASK_CAUSAL);
    DISPATCH_FA_INFER_LSE(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING,
                          half, half, FaiKernel::MaskType::MASK_CAUSAL);
    DISPATCH_FA_INFER(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_LOW_PREC_SPLITFUSE_TILING,
                      half, half, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER_LSE(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_LOW_PREC_SPLITFUSE_TILING,
                          half, half, FaiKernel::MaskType::NO_MASK);

    DISPATCH_FA_INFER(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING,
                      bfloat16_t, bfloat16_t, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER_LSE(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING,
                          bfloat16_t, bfloat16_t, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING,
                      bfloat16_t, bfloat16_t, FaiKernel::MaskType::MASK_CAUSAL);
    DISPATCH_FA_INFER_LSE(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING,
                          bfloat16_t, bfloat16_t, FaiKernel::MaskType::MASK_CAUSAL);
    DISPATCH_FA_INFER(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING,
                      bfloat16_t, bfloat16_t, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER_LSE(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING,
                          bfloat16_t, bfloat16_t, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING,
                      bfloat16_t, bfloat16_t, FaiKernel::MaskType::MASK_CAUSAL);
    DISPATCH_FA_INFER_LSE(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING,
                          bfloat16_t, bfloat16_t, FaiKernel::MaskType::MASK_CAUSAL);
    DISPATCH_FA_INFER(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_NOMASK_LOW_PREC_SPLITFUSE_TILING,
                      bfloat16_t, bfloat16_t, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER_LSE(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_NOMASK_LOW_PREC_SPLITFUSE_TILING,
                          bfloat16_t, bfloat16_t, FaiKernel::MaskType::NO_MASK);

    DISPATCH_FA_INFER_DECODE(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_DECODING_TILING,
                             half, half);
    DISPATCH_FA_INFER_DECODE(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_DECODING_TILING,
                             bfloat16_t, bfloat16_t);

    DISPATCH_FA_INFER_PAGED(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING,
                            half, half, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER_PAGED_LSE(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING,
                                half, half, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER_PAGED(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING,
                            half, half, FaiKernel::MaskType::MASK_CAUSAL);
    DISPATCH_FA_INFER_PAGED_LSE(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING,
                                half, half, FaiKernel::MaskType::MASK_CAUSAL);
    DISPATCH_FA_INFER_PAGED(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING,
                            bfloat16_t, bfloat16_t, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER_PAGED_LSE(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING,
                                bfloat16_t, bfloat16_t, FaiKernel::MaskType::NO_MASK);
    DISPATCH_FA_INFER_PAGED(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING,
                            bfloat16_t, bfloat16_t, FaiKernel::MaskType::MASK_CAUSAL);
    DISPATCH_FA_INFER_PAGED_LSE(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING,
                                bfloat16_t, bfloat16_t, FaiKernel::MaskType::MASK_CAUSAL);
}
