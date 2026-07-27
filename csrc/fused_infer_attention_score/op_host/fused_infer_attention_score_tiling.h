/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling.h
 * \brief
 */

#ifndef FUSED_INFER_ATTENTION_SCORE_TILING_H
#define FUSED_INFER_ATTENTION_SCORE_TILING_H

#include <cstdint>
#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/tiling_context.h>
#include <tiling/platform/platform_ascendc.h>
#include "register/tilingdata_base.h"
#include "flash_attention_infer_tiling.h"

namespace optiling {
// Inputs Index
constexpr uint32_t FIA_QUERY_INPUT_INDEX = 0;
constexpr uint32_t FIA_KEY_INPUT_INDEX = 1;
constexpr uint32_t FIA_VALUE_INPUT_INDEX = 2;
constexpr uint32_t FIA_PSE_SHIFT_INPUT_INDEX = 3;
constexpr uint32_t FIA_ATTEN_MASK_INPUT_INDEX = 4;
constexpr uint32_t FIA_ACTUAL_SEQ_LENGTHS_INPUT_INDEX = 5;
constexpr uint32_t FIA_ACTUAL_SEQ_LENGTHS_KV_INPUT_INDEX = 6;
constexpr uint32_t FIA_BLOCK_TABLE_INPUT_INDEX = 7;

// Outputs Index
constexpr uint32_t FIA_ATTENTION_OUT_INDEX = 0;
constexpr uint32_t FIA_SOFTMAX_LSE_INDEX = 1;

// Attributes Index
constexpr uint32_t FIA_NUM_HEADS_ATTR_INDEX = 0;
constexpr uint32_t FIA_SCALE_ATTR_INDEX = 1;
constexpr uint32_t FIA_PRE_TOKENS_ATTR_INDEX = 2;
constexpr uint32_t FIA_NEXT_TOKENS_ATTR_INDEX = 3;
constexpr uint32_t FIA_INPUT_LAYOUT_ATTR_INDEX = 4;
constexpr uint32_t FIA_NUM_KEY_VALUE_HEADS_ATTR_INDEX = 5;
constexpr uint32_t FIA_SPARSE_MODE_ATTR_INDEX = 6;
constexpr uint32_t FIA_INNER_PRECISE_ATTR_INDEX = 7;
constexpr uint32_t FIA_BLOCK_SIZE_ATTR_INDEX = 8;
constexpr uint32_t FIA_ANTIQUANT_MODE_ATTR_INDEX = 9;
constexpr uint32_t FIA_SPARSE_LAMBDA_ATTR_INDEX = 10;
constexpr uint32_t FIA_SOFTMAX_LSE_FLAG_ATTR_INDEX = 11;

ge::graphStatus TilingVllmFusedInferAttentionScore(gert::TilingContext *context);
} // namespace optiling

#endif // FUSED_INFER_ATTENTION_SCORE_TILING_H
