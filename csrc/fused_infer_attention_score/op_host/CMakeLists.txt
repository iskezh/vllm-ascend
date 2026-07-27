# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME VllmFusedInferAttentionScore
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

set(fused_infer_attention_score_depends transformer/attention/fused_infer_attention_score PARENT_SCOPE)
target_sources(op_host_aclnn PRIVATE
        vllm_fused_infer_attention_score_def.cpp
)

target_sources(optiling PRIVATE
        fused_infer_attention_score_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        fused_infer_attention_score_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        fused_infer_attention_score_proto.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# The tiling function copies optional device-side sequence-length tensors to
# host memory using the Ascend runtime API.
target_link_libraries(optiling PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64/libacl_rt.so
)
