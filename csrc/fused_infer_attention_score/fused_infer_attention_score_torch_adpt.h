/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FUSED_INFER_ATTENTION_SCORE_TORCH_ADPT_H
#define FUSED_INFER_ATTENTION_SCORE_TORCH_ADPT_H

#include <tuple>
#include <cstdio>
#include <torch/extension.h>

namespace vllm_ascend {

inline std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_score(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &blocktable,
    int64_t num_heads, double scale, int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, int64_t num_key_value_heads,
    int64_t sparse_mode, int64_t inner_precise, int64_t block_size,
    int64_t antiquant_mode, double sparse_lambda, bool softmax_lse_flag)
{
    TORCH_CHECK(query.dim() == 3, "query must be 3D TND layout");
    TORCH_CHECK(key.dim() == 3, "key must be 3D TND layout");
    TORCH_CHECK(value.dim() == 3, "value must be 3D TND layout");
    TORCH_CHECK(antiquant_mode == 0, "only antiquant_mode 0 is supported in this migration");

    at::Tensor attention_out = at::empty(query.sizes(), query.options().dtype(query.dtype()));
    at::Tensor softmax_lse = at::empty({query.size(0), query.size(1), 1}, query.options().dtype(at::kFloat));

    std::string input_layout_str = std::string(input_layout);
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    EXEC_NPU_CMD(
        aclnnVllmFusedInferAttentionScore,
        query,
        key,
        value,
        pse_shift,
        atten_mask,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        blocktable,
        num_heads,
        scale,
        pre_tokens,
        next_tokens,
        input_layout_ptr,
        num_key_value_heads,
        sparse_mode,
        inner_precise,
        block_size,
        antiquant_mode,
        sparse_lambda,
        softmax_lse_flag,
        attention_out,
        softmax_lse);

    return std::make_tuple(attention_out, softmax_lse);
}

} // namespace vllm_ascend

#endif // FUSED_INFER_ATTENTION_SCORE_TORCH_ADPT_H
