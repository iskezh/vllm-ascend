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
 * \file fused_infer_attention_score_tiling.cpp
 * \brief
 */

#include "fused_infer_attention_score_tiling.h"
#include "error/ops_error.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl_rt.h"
#include "exe_graph/runtime/tensor_data.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
// Dummy base tiling-data class for the op_type entry; SplitFuse kernels use
// FAInferTilingData registered per tiling key below.
BEGIN_TILING_DATA_DEF(VllmFusedInferAttentionScoreTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, reserved)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore, VllmFusedInferAttentionScoreTilingData)

// Register the single FAInferTilingData class for the supported tiling keys.
// Non-paged TND regular keys.
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000000200100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000000201100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000000200103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000000201103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000000200200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000000201200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000000200203, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000000201203, FAInferTilingData)
// Low-precision (inner_precise == 1) non-paged no-mask keys.
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000000210100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000000211100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000000210200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000000211200, FAInferTilingData)
// Paged-cache regular keys (paged + TND, no decoding/flash-decode).
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000010200100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000010201100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000010200103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000010201103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000010200200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000010201200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000010200203, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5000000000010201203, FAInferTilingData)
// Paged-cache strict-decode keys (qSeqlen==1, no-mask, no-LSE).
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5200000000010200100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmFusedInferAttentionScore_5200000000010200200, FAInferTilingData)

struct VllmFusedInferAttentionScoreCompileInfo {};

static ge::graphStatus CopySeqLengthsToHost(const gert::Tensor *tensor, std::vector<int64_t> &hostData)
{
    int64_t shapeSize = tensor->GetShapeSize();
    OPS_ERR_IF(shapeSize <= 0,
               OPS_LOG_E("VllmFusedInferAttentionScore", "invalid seq length tensor size"),
               return ge::GRAPH_FAILED);
    hostData.resize(static_cast<size_t>(shapeSize));
    auto placement = tensor->GetPlacement();
    const int64_t *src = tensor->GetData<int64_t>();
    size_t bytes = static_cast<size_t>(shapeSize) * sizeof(int64_t);
    if (gert::TensorPlacementUtils::IsOnHost(placement)) {
        std::memcpy(hostData.data(), src, bytes);
    } else {
        aclError ret = aclrtMemcpy(hostData.data(), bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        OPS_ERR_IF(ret != ACL_SUCCESS,
                   OPS_LOG_E("VllmFusedInferAttentionScore", "aclrtMemcpy for seq lengths failed"),
                   return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForVllmFusedInferAttentionScore(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertContextToFAInferContext(gert::TilingContext *context, FAInferContext &faInfo,
                                                      const std::vector<int64_t> &hostActualQSeq,
                                                      const std::vector<int64_t> &hostActualKvSeq)
{
    auto qDesc = context->GetInputDesc(FIA_QUERY_INPUT_INDEX);
    OPS_ERR_IF(qDesc == nullptr, OPS_LOG_E("VllmFusedInferAttentionScore", "query desc is nullptr"),
               return ge::GRAPH_FAILED);
    ge::DataType qDataType = qDesc->GetDataType();
    OPS_ERR_IF(qDataType != ge::DT_FLOAT16 && qDataType != ge::DT_BF16,
               OPS_LOG_E("VllmFusedInferAttentionScore", "query dtype must be FP16 or BF16"),
               return ge::GRAPH_FAILED);

    auto kDesc = context->GetInputDesc(FIA_KEY_INPUT_INDEX);
    auto vDesc = context->GetInputDesc(FIA_VALUE_INPUT_INDEX);
    OPS_ERR_IF(kDesc == nullptr || vDesc == nullptr,
               OPS_LOG_E("VllmFusedInferAttentionScore", "key/value desc is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kDesc->GetDataType() != qDataType || vDesc->GetDataType() != qDataType,
               OPS_LOG_E("VllmFusedInferAttentionScore", "key/value dtype must match query dtype"),
               return ge::GRAPH_FAILED);

    auto qShape = context->GetInputShape(FIA_QUERY_INPUT_INDEX);
    auto kShape = context->GetInputShape(FIA_KEY_INPUT_INDEX);
    auto vShape = context->GetInputShape(FIA_VALUE_INPUT_INDEX);
    OPS_ERR_IF(qShape == nullptr || kShape == nullptr || vShape == nullptr,
               OPS_LOG_E("VllmFusedInferAttentionScore", "input shape is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(qShape->GetStorageShape().GetDimNum() != 3,
               OPS_LOG_E("VllmFusedInferAttentionScore", "query must be 3D TND layout"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kShape->GetStorageShape().GetDimNum() != 3 || vShape->GetStorageShape().GetDimNum() != 3,
               OPS_LOG_E("VllmFusedInferAttentionScore", "key/value must be 3D TND layout"),
               return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E("VllmFusedInferAttentionScore", "attrs is nullptr"),
               return ge::GRAPH_FAILED);

    const char *layoutPtr = attrs->GetAttrPointer<char>(FIA_INPUT_LAYOUT_ATTR_INDEX);
    OPS_ERR_IF(layoutPtr == nullptr, OPS_LOG_E("VllmFusedInferAttentionScore", "input_layout attr is nullptr"),
               return ge::GRAPH_FAILED);
    std::string layoutStr(layoutPtr);
    OPS_ERR_IF(layoutStr != "TND",
               OPS_LOG_E("VllmFusedInferAttentionScore", "only TND layout is supported"),
               return ge::GRAPH_FAILED);

    const int64_t *numHeadsPtr = attrs->GetAttrPointer<int64_t>(FIA_NUM_HEADS_ATTR_INDEX);
    const int64_t *numKvHeadsPtr = attrs->GetAttrPointer<int64_t>(FIA_NUM_KEY_VALUE_HEADS_ATTR_INDEX);
    OPS_ERR_IF(numHeadsPtr == nullptr || numKvHeadsPtr == nullptr,
               OPS_LOG_E("VllmFusedInferAttentionScore", "num_heads/num_key_value_heads attr is nullptr"),
               return ge::GRAPH_FAILED);
    int64_t numHeads = *numHeadsPtr;
    int64_t numKvHeads = (*numKvHeadsPtr == 0) ? numHeads : *numKvHeadsPtr;
    OPS_ERR_IF(numHeads == 0 || numKvHeads == 0 || numHeads % numKvHeads != 0,
               OPS_LOG_E("VllmFusedInferAttentionScore", "invalid num_heads/num_key_value_heads"),
               return ge::GRAPH_FAILED);

    const int64_t *sparseModePtr = attrs->GetAttrPointer<int64_t>(FIA_SPARSE_MODE_ATTR_INDEX);
    OPS_ERR_IF(sparseModePtr == nullptr,
               OPS_LOG_E("VllmFusedInferAttentionScore", "sparse_mode attr is nullptr"),
               return ge::GRAPH_FAILED);
    int64_t sparseMode = *sparseModePtr;
    OPS_ERR_IF(sparseMode != 0 && sparseMode != 3,
               OPS_LOG_E("VllmFusedInferAttentionScore", "only sparse_mode 0 or 3 is supported"),
               return ge::GRAPH_FAILED);

    // pse_shift is not supported in this minimal migration
    OPS_ERR_IF(context->GetOptionalInputShape(FIA_PSE_SHIFT_INPUT_INDEX) != nullptr,
               OPS_LOG_E("VllmFusedInferAttentionScore", "pse_shift is not supported"),
               return ge::GRAPH_FAILED);

    const int64_t *antiquantModePtr = attrs->GetAttrPointer<int64_t>(FIA_ANTIQUANT_MODE_ATTR_INDEX);
    OPS_ERR_IF(antiquantModePtr == nullptr,
               OPS_LOG_E("VllmFusedInferAttentionScore", "antiquant_mode attr is nullptr"),
               return ge::GRAPH_FAILED);
    int64_t antiquantMode = *antiquantModePtr;
    OPS_ERR_IF(antiquantMode != 0,
               OPS_LOG_E("VllmFusedInferAttentionScore", "only antiquant_mode 0 is supported in this migration"),
               return ge::GRAPH_FAILED);

    const float *sparseLambdaPtr = attrs->GetAttrPointer<float>(FIA_SPARSE_LAMBDA_ATTR_INDEX);
    OPS_ERR_IF(sparseLambdaPtr == nullptr,
               OPS_LOG_E("VllmFusedInferAttentionScore", "sparse_lambda attr is nullptr"),
               return ge::GRAPH_FAILED);
    float sparseLamda = *sparseLambdaPtr;

    const int64_t *preTokenPtr = attrs->GetAttrPointer<int64_t>(FIA_PRE_TOKENS_ATTR_INDEX);
    const int64_t *nextTokenPtr = attrs->GetAttrPointer<int64_t>(FIA_NEXT_TOKENS_ATTR_INDEX);
    OPS_ERR_IF(preTokenPtr == nullptr || nextTokenPtr == nullptr,
               OPS_LOG_E("VllmFusedInferAttentionScore", "pre_tokens/next_tokens attr is nullptr"),
               return ge::GRAPH_FAILED);
    int64_t preToken = *preTokenPtr;
    int64_t nextToken = *nextTokenPtr;
    if (preToken > SPARSE_MODE_INT_MAX) {
        preToken = SPARSE_MODE_INT_MAX;
    } else if (preToken < -SPARSE_MODE_INT_MAX) {
        preToken = -SPARSE_MODE_INT_MAX;
    }
    if (nextToken > SPARSE_MODE_INT_MAX) {
        nextToken = SPARSE_MODE_INT_MAX;
    } else if (nextToken < -SPARSE_MODE_INT_MAX) {
        nextToken = -SPARSE_MODE_INT_MAX;
    }

    const float *scalePtr = attrs->GetAttrPointer<float>(FIA_SCALE_ATTR_INDEX);
    OPS_ERR_IF(scalePtr == nullptr, OPS_LOG_E("VllmFusedInferAttentionScore", "scale attr is nullptr"),
               return ge::GRAPH_FAILED);

    const int64_t *blockSizePtr = attrs->GetAttrPointer<int64_t>(FIA_BLOCK_SIZE_ATTR_INDEX);
    const int64_t *innerPrecisePtr = attrs->GetAttrPointer<int64_t>(FIA_INNER_PRECISE_ATTR_INDEX);
    const bool *lseFlagPtr = attrs->GetAttrPointer<bool>(FIA_SOFTMAX_LSE_FLAG_ATTR_INDEX);
    OPS_ERR_IF(blockSizePtr == nullptr || innerPrecisePtr == nullptr || lseFlagPtr == nullptr,
               OPS_LOG_E("VllmFusedInferAttentionScore", "block_size/inner_precise/softmax_lse_flag attr is nullptr"),
               return ge::GRAPH_FAILED);
    int64_t blockSize = *blockSizePtr;
    int64_t innerPrecise = *innerPrecisePtr;
    bool lseFlag = *lseFlagPtr;

    OPS_ERR_IF(innerPrecise != 0,
               OPS_LOG_E("VllmFusedInferAttentionScore", "only inner_precise 0 is supported"),
               return ge::GRAPH_FAILED);

    auto actualQSeq = context->GetOptionalInputTensor(FIA_ACTUAL_SEQ_LENGTHS_INPUT_INDEX);
    auto actualKvSeq = context->GetOptionalInputTensor(FIA_ACTUAL_SEQ_LENGTHS_KV_INPUT_INDEX);
    OPS_ERR_IF(actualQSeq == nullptr || actualKvSeq == nullptr,
               OPS_LOG_E("VllmFusedInferAttentionScore", "actual_seq_lengths and actual_seq_lengths_kv are required"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(actualQSeq->GetDataType() != ge::DT_INT64 || actualKvSeq->GetDataType() != ge::DT_INT64,
               OPS_LOG_E("VllmFusedInferAttentionScore", "actual_seq_lengths must be INT64"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(hostActualQSeq.empty() || hostActualKvSeq.empty(),
               OPS_LOG_E("VllmFusedInferAttentionScore", "actual_seq_lengths host data is empty"),
               return ge::GRAPH_FAILED);
    const int64_t *actualSeqQ = hostActualQSeq.data();
    const int64_t *actualSeqKv = hostActualKvSeq.data();
    int32_t batch = static_cast<int32_t>(hostActualQSeq.size());
    OPS_ERR_IF(batch <= 0,
               OPS_LOG_E("VllmFusedInferAttentionScore", "invalid actual_seq_lengths size"),
               return ge::GRAPH_FAILED);

    auto blockTableShape = context->GetOptionalInputShape(FIA_BLOCK_TABLE_INPUT_INDEX);
    bool pagedCacheFlag = (blockTableShape != nullptr);
    if (pagedCacheFlag) {
        OPS_ERR_IF(blockTableShape->GetStorageShape().GetDimNum() != 2,
                   OPS_LOG_E("VllmFusedInferAttentionScore", "block_table must be 2D"),
                   return ge::GRAPH_FAILED);
        auto blockTableDesc = context->GetOptionalInputDesc(FIA_BLOCK_TABLE_INPUT_INDEX);
        OPS_ERR_IF(blockTableDesc != nullptr && blockTableDesc->GetDataType() != ge::DT_INT32,
                   OPS_LOG_E("VllmFusedInferAttentionScore", "block_table must be INT32"),
                   return ge::GRAPH_FAILED);
    }

    faInfo.pagedCacheFlag = pagedCacheFlag;
    faInfo.numHeads = static_cast<int32_t>(numHeads);
    faInfo.kvHeads = static_cast<int32_t>(numKvHeads);
    faInfo.numBlocks = static_cast<int32_t>(kShape->GetStorageShape().GetDim(0));
    faInfo.blockSize = static_cast<int32_t>(blockSize);
    faInfo.embeddingSize = static_cast<int32_t>(qShape->GetStorageShape().GetDim(2));
    faInfo.embeddingSizeV = faInfo.embeddingSize;
    faInfo.scaleValue = *scalePtr;
    faInfo.sparseLamda = sparseLamda;
    faInfo.sparseMode = static_cast<int32_t>(sparseMode);
    faInfo.preToken = preToken;
    faInfo.nextToken = nextToken;
    faInfo.layout = layoutStr;
    faInfo.lseFlag = lseFlag;
    faInfo.learnableSinkFlag = false;
    faInfo.innerPrecise = static_cast<int32_t>(innerPrecise);
    faInfo.dataType = (qDataType == ge::DT_BF16) ? DataType::BF16 : DataType::FP16;
    faInfo.batch = batch;
    faInfo.qSeqlenList = actualSeqQ;
    faInfo.kvSeqlenList = actualSeqKv;
    faInfo.isTilingSink = false;
    faInfo.maskType = (sparseMode == 3) ? MaskType::MASK_SPEC : MaskType::NO_MASK;
    faInfo.pseQ = 0;
    faInfo.pseKv = 0;
    if (pagedCacheFlag) {
        faInfo.maxNumBlocksPerBatch = static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(1));
    }

    int64_t maxQSeqlen = 0;
    int64_t minQSeqlen = INT64_MAX;
    int64_t maxKvSeqlen = 0;
    int64_t minKvSeqlen = INT64_MAX;
    for (int32_t b = 0; b < batch; ++b) {
        int64_t qSeqlen = actualSeqQ[b];
        int64_t kvSeqlen = actualSeqKv[b];
        if (b > 0) {
            qSeqlen -= actualSeqQ[b - 1];
            if (!pagedCacheFlag) {
                kvSeqlen -= actualSeqKv[b - 1];
            }
        }
        maxQSeqlen = std::max(maxQSeqlen, qSeqlen);
        minQSeqlen = std::min(minQSeqlen, qSeqlen);
        maxKvSeqlen = std::max(maxKvSeqlen, kvSeqlen);
        minKvSeqlen = std::min(minKvSeqlen, kvSeqlen);
    }
    faInfo.maxQSeqlen = static_cast<int64_t>(maxQSeqlen);
    faInfo.maxKvSeqlen = static_cast<int64_t>(maxKvSeqlen);

    faInfo.flashDecodeFlag = false;
    faInfo.decodingFlag = false;
    // The source warehouse does not hard-reject paged-cache inputs.  Allow all
    // paged-cache cases to fall through to the generic FAInferTiling path, and
    // only select the dedicated decoding kernel for the strict qSeqlen==1 decode
    // case (matching the pre-migration fast-path).
    if (pagedCacheFlag && maxQSeqlen == 1 && minQSeqlen == 1 &&
        faInfo.maskType == MaskType::NO_MASK && !lseFlag) {
        faInfo.decodingFlag = true;
    }

    faInfo.workspaces = context->GetWorkspaceSizes(1);

    {
        constexpr uint64_t SPLIT_FUSE_BASE_KEY = 5000000000000000000ULL;
        constexpr uint64_t PAGED_CACHE_KEY = 10000000ULL;
        constexpr uint64_t COMP_CAUSAL_MASK_KEY = 3;
        constexpr uint64_t COMP_SWA_MASK_KEY = 5;
        constexpr uint64_t FULL_MASK_KEY = 6;
        constexpr uint64_t LAYOUTQ_TND_KEY = 200000;
        constexpr uint64_t DTYPE_FP16_KEY = 100;
        constexpr uint64_t DTYPE_BF16_KEY = 200;
        constexpr uint64_t LSE_OUT_ONLY_KEY = 1000;
        constexpr uint64_t FLASH_DECODE_KEY = 100000000000000000ULL;
        constexpr uint64_t DECODING_KEY = 200000000000000000ULL;
        uint64_t key = SPLIT_FUSE_BASE_KEY;
        if (faInfo.pagedCacheFlag) key += PAGED_CACHE_KEY;
        if (faInfo.maskType == MaskType::MASK_SPEC) key += COMP_CAUSAL_MASK_KEY;
        else if (faInfo.maskType == MaskType::SWA_MASK) key += COMP_SWA_MASK_KEY;
        else if (faInfo.maskType == MaskType::FULL_MASK) key += FULL_MASK_KEY;
        if (faInfo.layout == "TND") key += LAYOUTQ_TND_KEY;
        if (faInfo.dataType == DataType::FP16) key += DTYPE_FP16_KEY;
        else if (faInfo.dataType == DataType::BF16) key += DTYPE_BF16_KEY;
        if (faInfo.lseFlag) key += LSE_OUT_ONLY_KEY;
        if (faInfo.flashDecodeFlag) key += FLASH_DECODE_KEY;
        else if (faInfo.decodingFlag) key += DECODING_KEY;
        OPS_LOG_I(context->GetNodeName(),
                  "FIA debug: key=%lu paged=%d decode=%d flashDecode=%d mask=%d lse=%d "
                  "maxQ=%ld minQ=%ld minKv=%ld layout=%s dtype=%d sparseLambda=%.2f",
                  key, faInfo.pagedCacheFlag, faInfo.decodingFlag, faInfo.flashDecodeFlag,
                  static_cast<int>(faInfo.maskType), faInfo.lseFlag, maxQSeqlen, minQSeqlen, minKvSeqlen,
                  faInfo.layout.c_str(), static_cast<int>(faInfo.dataType), sparseLamda);
        for (int32_t b = 0; b < batch; ++b) {
            int64_t ql = actualSeqQ[b];
            int64_t kvl = actualSeqKv[b];
            if (b > 0) {
                ql -= actualSeqQ[b - 1];
                if (!pagedCacheFlag) kvl -= actualSeqKv[b - 1];
            }
            printf("[FIA_TILING_DBG] batch[%d] qlen=%ld kvlen=%ld\n", b, (long)ql, (long)kvl);
        }
        printf("[FIA_TILING_DBG] sparseLambda=%.2f sparseMode=%ld paged=%d blockSize=%ld "
                "numBlocks=%d maxQ=%ld\n",
                sparseLamda, (long)sparseMode, (int)pagedCacheFlag, (long)blockSize,
                (int)(kShape->GetStorageShape().GetDim(0)), (long)maxQSeqlen);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingVllmFusedInferAttentionScore(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("VllmFusedInferAttentionScore", "TilingContext is nullptr"),
               return ge::GRAPH_FAILED);

    auto platformInfoPtr = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfoPtr == nullptr,
               OPS_LOG_E("VllmFusedInferAttentionScore", "PlatformInfo is nullptr"),
               return ge::GRAPH_FAILED);

    // Use batch mode to ensure all cores start together.
    context->SetScheduleMode(1);

    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfoPtr);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAic();

    FAInferContext faInfo;
    std::vector<int64_t> hostActualQSeq;
    std::vector<int64_t> hostActualKvSeq;
    auto actualQSeqTensor = context->GetOptionalInputTensor(FIA_ACTUAL_SEQ_LENGTHS_INPUT_INDEX);
    auto actualKvSeqTensor = context->GetOptionalInputTensor(FIA_ACTUAL_SEQ_LENGTHS_KV_INPUT_INDEX);
    OPS_ERR_IF(actualQSeqTensor == nullptr || actualKvSeqTensor == nullptr,
               OPS_LOG_E("VllmFusedInferAttentionScore", "actual_seq_lengths tensors are required"),
               return ge::GRAPH_FAILED);
    auto ret = CopySeqLengthsToHost(actualQSeqTensor, hostActualQSeq);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CopySeqLengthsToHost(actualKvSeqTensor, hostActualKvSeq);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ConvertContextToFAInferContext(context, faInfo, hostActualQSeq, hostActualKvSeq);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    FAInferTiling faTiling(faInfo);
    faTiling.SetCoreNum(coreNum);

    FAInferTilingData faTilingData;
    ret = faTiling.DoTiling(faTilingData);
    OPS_ERR_IF(ret != ge::GRAPH_SUCCESS,
               OPS_LOG_E("VllmFusedInferAttentionScore", "FAInferTiling DoTiling failed"),
               return ge::GRAPH_FAILED);

    printf("[FIA_TILING_POST] coreNum=%u totalTaskNum=%u firstBatchTaskNum=%u "
            "maxNumBlocksPerBatch=%u sparseLambda=%.2f\n",
            coreNum, faTilingData.get_totalTaskNum(), faTilingData.get_firstBatchTaskNum(),
            (uint32_t)faTilingData.get_maxNumBlocksPerBatch(), faTilingData.get_sparseLamda());

    faTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(faTilingData.GetDataSize());

    size_t *workspaces = context->GetWorkspaceSizes(1);
    OPS_ERR_IF(workspaces == nullptr,
               OPS_LOG_E("VllmFusedInferAttentionScore", "workspace sizes array is nullptr"),
               return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(16U * 1024U * 1024U) +
                    static_cast<size_t>(faTiling.GetCoreNum()) * WORKSPACE_BLOCK_SIZE_DB * 4U * 3U * 4U +
                    static_cast<size_t>(faTilingData.get_splitLseTotalSize()) +
                    static_cast<size_t>(faTilingData.get_splitOTotalSize());

    context->SetBlockDim(faTiling.GetCoreNum());
    context->SetTilingKey(faTiling.GetTilingKey());

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(VllmFusedInferAttentionScore)
    .TilingInputsDataDependency({FIA_ACTUAL_SEQ_LENGTHS_INPUT_INDEX, FIA_ACTUAL_SEQ_LENGTHS_KV_INPUT_INDEX},
                                {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_HOST})
    .Tiling(TilingVllmFusedInferAttentionScore)
    .TilingParse<VllmFusedInferAttentionScoreCompileInfo>(TilingPrepareForVllmFusedInferAttentionScore);
} // namespace optiling
