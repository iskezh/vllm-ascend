import torch
import torch_npu

import ctypes
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray
from atk.tasks.backends.lib_interface.acl_wrapper import VoidPtr
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor, nnopbase, AclFormat
import logging
import numpy as np
from array import array
import random
from ml_dtypes import bfloat16
from dataclasses import dataclass

# np.random.seed(44)
IS_INF_FLAG=False

dtypeMap = {
    torch.float16: np.float16,
    torch.bfloat16: bfloat16,
    torch.float32: np.float32
}

maskTypeMap = {
    ## sparseMode : golden maskType
    0: 0,
    3: 1,
    4: 2
}

class TestFIAV4SplitFuse():
    @dataclass
    class AuxAttrs:
        preTokens: int
        nextTokens: int
        num_heads: int
        kv_heads: int
        head_size: int
        num_blocks: int
        block_size: int
        mask_type: int
        dtype: any
        kv_dtype: int
        layout_dtype: int
        max_q_seqlen: int
        max_kv_seqlen: int
        inner_prec: int
        scale: float
        sparseMode: int

    
    @dataclass
    class AttentionInputs:
        query: any
        key_cache: any
        value_cache: any
        block_tables: any
        q_seqlen_list: any
        k_seqlen_list: any
        global_mask: any
        learnable_sink: any
        auxAttrs: any

    @classmethod
    def group_matmul(cls, head, kv_head, left, right, high_prec = 1):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            if high_prec == 0:
                group_score = np.matmul(left[i * group_num:(i + 1) * group_num, :, :],
                                        right[i:(i + 1), :, :]).astype(np.float32)
            else:
                group_score = np.matmul(left[i * group_num:(i + 1) * group_num, :, :].astype(np.float32),
                                        right[i:(i + 1), :, :].astype(np.float32)).astype(np.float32)
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        return score

    @classmethod
    def softmax_numpy(cls, sim, sink_matrix, batch_i):
        row_max = np.max(sim, axis=-1, keepdims=True)
        valid_row_mask = ~np.isneginf(row_max)
        # add sink rowmax
        if sink_matrix is not None:
            assert sink_matrix.shape == row_max.shape, \
                f"sink_matrix 形状 {sink_matrix.shape} 与 row_max 形状 {row_max.shape} 不一致！"
            # 更新含sink的rowmax
            # row_max = np.maximum(row_max, sink_matrix)
            row_max[valid_row_mask] = np.maximum(
                row_max[valid_row_mask], 
                sink_matrix[valid_row_mask]
            )
        
        sim_sub = sim - row_max
        sim_sub_high = sim.astype(np.float64) - row_max.astype(np.float64)

        sim_sub = np.exp(sim_sub)
        sim_sub_high = np.exp(sim_sub_high)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        row_sum_high = np.sum(sim_sub_high, axis=-1, keepdims=True)

        if sink_matrix is not None:
            sink_exp = np.exp(sink_matrix - row_max)
            sink_exp_high = np.exp(sink_matrix.astype(np.float64) - row_max.astype(np.float64))
            row_sum = row_sum + sink_exp
            row_sum_high = row_sum_high + sink_exp_high

        soft_res = sim_sub / row_sum
        lse = np.squeeze((np.log(row_sum_high) + row_max.astype(np.float64)), axis=-1)
        # lse = np.squeeze((np.log(row_sum) + row_max), axis=-1)

        return soft_res, lse, row_max

    def softmax1(
        self,
        qk_result,
        is_first,
        gm,
        is_kvs_last_loop,
        sink_matrix,
        interm_dtype = np.float16
    ):
        sim = qk_result.astype(interm_dtype)
        # sim = qk_result
        lm = np.max(sim, axis=-1, keepdims=True)
        if is_first:
            hm = lm
            dm = 0

        else:
            hm = np.maximum(gm, lm)
            dm = gm - hm
        
        valid_hm_mask = ~np.isneginf(hm)
        if sink_matrix is not None and is_kvs_last_loop:
            assert sink_matrix.shape == hm.shape, \
            f"sink_matrix 形状 {sink_matrix.shape} 与 hm 形状 {hm.shape} 不一致！"
            hm[valid_hm_mask] = np.maximum(
                hm[valid_hm_mask], 
                sink_matrix[valid_hm_mask]
            )
            # hm = np.maximum(hm, sink_matrix)
            dm = gm - hm if not is_first else 0

        gm = hm
        sim_sub = sim - hm
        sim_sub = np.exp(sim_sub.astype(interm_dtype))
        # sim_sub = sim_sub.astype(np.float16)

        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)

        sink_exp = None
        if sink_matrix is not None and is_kvs_last_loop:
            sink_exp = np.exp((sink_matrix - hm).astype(interm_dtype)).astype(interm_dtype)
            # row_sum = row_sum + sink_exp

        return sim_sub, row_sum, dm, gm, sink_exp


    def qkMM1(
        self,
        query,
        key
    ):
        result = None
        qk_k = key.shape[1]
        qk_k_split = 128
        qk_k_loop = (qk_k + 127) // 128
        for qk_k_loop_idx in range(qk_k_loop):
            sub_k = 128 if qk_k_loop_idx != (qk_k_loop - 1) else (qk_k - qk_k_loop_idx * 128)
            partial_Query = query[:, :, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k]
            partial_Key = key[:, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k, :]
            result_split = self.group_matmul(partial_Query.shape[0], partial_Key.shape[0], partial_Query, partial_Key, 0)
            if result is None:
                result = result_split
            else:
                result = result + result_split
        return result
    
    def pvMM2(
        self,
        p,
        value
    ):
        result = None
        pv_k = value.shape[1]
        pv_k_split = 128
        pv_k_loop = (pv_k + 127) // 128
        for pv_k_loop_idx in range(pv_k_loop):
            sub_k = 128 if pv_k_loop_idx != (pv_k_loop - 1) else (pv_k - pv_k_loop_idx * 128)
            partial_P = p[:, :, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k]
            partial_Value = value[:, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k, :]
            result_split = self.group_matmul(partial_P.shape[0], partial_Value.shape[0], partial_P, partial_Value, 0)
            if result is None:
                result = result_split
            else:
                result = result + result_split
        return result

    def ref_flash_attention(
        self,
        query,
        key,
        value,
        scale,
        mask,
        attention_inputs: AttentionInputs,
        sink_matrix,
        batch_i
    ):
        data_type = attention_inputs.auxAttrs.dtype
        inner_prec = attention_inputs.auxAttrs.inner_prec
        interm_dtype = np.float16 if inner_prec == 1 else np.float32
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        value = np.transpose(value, (1, 0, 2))
        scale = np.float16(scale) if inner_prec == 1 else np.float32(scale)
        context_len = key.shape[2]
        context_size = 512
        group_num = query.shape[0] // key.shape[0]
        gl = None
        gl_high = None
        go = None
        go_high = None
        gm = None

        if batch_i == 1:
            print(f"lwg-query.shape: {query.shape} \n")
            print(f"lwg-query.type----------------: {data_type} \n")

        is_kvs_last_loop = False

        for kv_start in range(0, context_len, context_size):
            sub_len = context_size
            if kv_start + context_size > context_len:
                sub_len = context_len - kv_start
            
            is_kvs_last_loop = (kv_start + context_size >= context_len)

            sub_key = key[:, :, kv_start: kv_start + sub_len]
            sub_mask = None
            if mask is not None:
                sub_mask = mask[:query.shape[1], kv_start: kv_start + sub_len].astype(interm_dtype)
            sub_value = value[:, kv_start: kv_start + sub_len, :]
            qk_result = self.qkMM1(query, sub_key).astype(interm_dtype)
            qk_result = qk_result * scale

            if mask is not None:
                qk_result += sub_mask
            if kv_start == 0:
                gm = None
            p_result, row_sum, dm, gm, sink_exp = self.softmax1(qk_result, kv_start == 0, gm, is_kvs_last_loop, sink_matrix, interm_dtype)
            p_result = p_result.astype(data_type)
            if kv_start == 0:
                gm_high = None
            lo = self.pvMM2(p_result, sub_value).astype(interm_dtype)
            
            if kv_start == 0:
                gl = row_sum
                go = lo
            else:
                dm = np.exp(dm)
                gl = gl * dm
                gl = gl + row_sum
                go = go * dm
                go = go + lo

            if is_kvs_last_loop and sink_exp is not None:
                assert gl.shape == sink_exp.shape, \
                f"sink_matrix 形状 {gl.shape} 与 hm 形状 {sink_exp.shape} 不一致！"
                print(f"lwg-+sink_exp \n")
                gl = gl + sink_exp

        go = go / gl
        go = np.transpose(go, (1, 0, 2))
        lse = np.squeeze((np.log(gl) + gm), axis=-1).astype(np.float32) # lse仅支持fp32输出，无论采取何种精度运算，最终结果都是fp32
        # if batch_i == 1:
        #     print(f"gm.shape: {gm.shape}\n")
        #     print(f"gm.value: {gm[2, 0:128, 0]}\n")
        #     print(f"gl: {gl}\n")
        #     print(f"gl.value: {gl[2, 0:128, 0]}\n")
        return go.astype(data_type), lse

    def ref_masked_attention(self,
            query,  # (q_seqlen, num_heads, head_size)
            key,    # (k_seqlen, kv_heads, head_size)
            value,
            scale: float,
            mask,    # (q_seqlen, k_seqlen)
            sink_matrix,
            batch_i
    ):
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        sim_high = self.group_matmul(query.shape[0], key.shape[0], query, key, 1)  # (head_num, q_seqlen, k_seqlen)
        sim_high = sim_high * scale
        if mask is not None:
            sim_high = sim_high + (
                mask[:sim_high.shape[-2], :sim_high.shape[-1]]
                ).astype(np.float32)
        p_high, lse_high, gm = self.softmax_numpy(sim_high, sink_matrix, batch_i)
        lse_high = lse_high.astype(np.float64)
        p = p_high.astype(query.dtype)
        p_high = p_high.astype(np.float32)
        value = np.transpose(value, (1, 0, 2))
        
        out_high = self.group_matmul(query.shape[0], key.shape[0], p_high, value, 1)
        out_high = np.transpose(out_high, (1, 0, 2))
        return out_high, lse_high

    def ref_single_query_cached_kv_attention(self, attention_inputs: AttentionInputs, output, golden_gpu_output, golden_lse_output, golden_gpu_lse_output) -> None:
        num_heads = attention_inputs.auxAttrs.num_heads
        kv_heads = attention_inputs.auxAttrs.kv_heads
        head_size_qk = attention_inputs.auxAttrs.head_size
        head_size_vo = attention_inputs.auxAttrs.head_size
        block_size = attention_inputs.auxAttrs.block_size
        max_q_seqlen = attention_inputs.auxAttrs.max_q_seqlen
        inner_prec = attention_inputs.auxAttrs.inner_prec
        scale = attention_inputs.auxAttrs.scale
        learnable_sink = attention_inputs.learnable_sink
        sparseMode = attention_inputs.auxAttrs.sparseMode
        if learnable_sink is not None:
            learnable_sink = learnable_sink.astype(np.float32)
            # print("sink_lwg")

        batch = len(attention_inputs.q_seqlen_list)
        cu_seqlen = 0
        kv_seqlen_now = 0
        for i in range(batch):
            q_seqlen = int(attention_inputs.q_seqlen_list[i])
            k_seqlen = int(attention_inputs.k_seqlen_list[i])
            q = None
            if attention_inputs.auxAttrs.layout_dtype == 1:
                q = attention_inputs.query[cu_seqlen:(cu_seqlen + q_seqlen), :, :]
            else:
                q = attention_inputs.query[i * max_q_seqlen:(i * max_q_seqlen + q_seqlen), :, :]
            keys = None
            values = None
            if attention_inputs.auxAttrs.kv_dtype == 1:
                keys = []
                values = []
                block_table = attention_inputs.block_tables[i]
                for j in range(k_seqlen):
                    if block_size == 0:
                        print(f"lwg-block_size: {block_size} \n")
                    block_number = int(block_table[j // block_size])
                    block_offset = j % block_size

                    k = attention_inputs.key_cache[block_number, block_offset, :, :]
                    k = k.reshape(kv_heads, head_size_qk)
                    keys.append(k)

                    v = attention_inputs.value_cache[block_number, block_offset, :, :]
                    v = v.reshape(kv_heads, head_size_vo)
                    values.append(v)
                keys = np.stack(keys, axis=0)
                values = np.stack(values, axis=0)
            elif attention_inputs.auxAttrs.kv_dtype == 0:
                if attention_inputs.auxAttrs.layout_dtype == 1:
                    keys = attention_inputs.key_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
                    values = attention_inputs.value_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
                else:
                    keys = attention_inputs.key_cache[i, :, :, :]
                    values = attention_inputs.value_cache[i, :, :, :]
            
            if attention_inputs.auxAttrs.mask_type == 1:
                mask = attention_inputs.global_mask[cu_seqlen:(cu_seqlen + q_seqlen), :]
            elif attention_inputs.auxAttrs.mask_type == 2:
                mask = attention_inputs.global_mask[cu_seqlen:(cu_seqlen + q_seqlen), :]
                print(f"ljl-batch:{i},attention_inputs.global_mask:{attention_inputs.global_mask.shape},{cu_seqlen},{q_seqlen}")
            elif attention_inputs.auxAttrs.mask_type == 0:
                mask = None

            sink_matrix = None
            if learnable_sink is not None:
                # [num_heads → [num_heads, 1, 1]
                sink_expanded = np.expand_dims(learnable_sink, axis=1)
                sink_expanded = np.expand_dims(sink_expanded, axis=2)
                # [num_heads, 1, 1] → [num_heads, q_seqlen, 1]
                sink_matrix = np.broadcast_to(sink_expanded, shape=(learnable_sink.shape[0], q_seqlen, 1))

            # # ljl
            preTokens = attention_inputs.auxAttrs.preTokens
            nextTokens = attention_inputs.auxAttrs.nextTokens
            
            preTokensChange = preTokens - k_seqlen + q_seqlen
            nextTokensChange = nextTokens + k_seqlen - q_seqlen
            nextTokensError = -nextTokensChange if nextTokensChange < 0 else 0
            preTokensError = (q_seqlen - k_seqlen - preTokensChange) if q_seqlen > k_seqlen + preTokensChange else 0
            actualSeq = q_seqlen
            print(f"ljl-2 {i},:{preTokens},{nextTokens},{preTokensChange},{nextTokensChange},{preTokensError},{nextTokensError}")
            # nextTokensChange += nextTokensError
            # preTokensChange -= nextTokensError
            actualSeq -= nextTokensError
            actualSeq -= preTokensError
            if actualSeq != q_seqlen and sparseMode == 4:
                if nextTokensError != 0:
                    # 前n行置0
                    actualSeq = q_seqlen - actualSeq
                elif preTokensError != 0:
                    # 后n行置0
                    actualSeq = actualSeq
            
            out_normal, lse = self.ref_masked_attention(q, keys, values, scale, mask, sink_matrix, i)
            # out_normal, lse = self.ref_flash_attention(q, keys, values, scale, mask, attention_inputs, sink_matrix, i)
            out_gpu, lse_gpu = self.ref_flash_attention(q, keys, values, scale, mask, attention_inputs, sink_matrix, i)
                        # out_normal, lse = out_gpu.astype(np.float32), lse_gpu
            out_gpu_test = torch.from_numpy(out_gpu.astype(np.float32))
            nan_out_gpu = torch.isnan(out_gpu_test)
            nan_count = nan_out_gpu.sum().item()

            out = out_normal.reshape(-1, num_heads, head_size_vo)
            out = out.reshape(-1, num_heads, head_size_vo)
            out_gpu = out_gpu.reshape(-1, num_heads, head_size_vo)

            if attention_inputs.auxAttrs.layout_dtype == 1:
                output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out
                golden_gpu_output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out_gpu

                golden_lse_output[:, cu_seqlen: cu_seqlen + q_seqlen] = lse
                golden_gpu_lse_output[:, cu_seqlen: cu_seqlen + q_seqlen] = lse_gpu
                
                if actualSeq != q_seqlen and sparseMode == 4:
                    if nextTokensError != 0:
                        output[cu_seqlen : cu_seqlen  + actualSeq, :, :] = 0  # 前n行置0
                        golden_gpu_output[cu_seqlen: cu_seqlen + actualSeq, :, :] = 0
                        golden_lse_output[:, cu_seqlen: cu_seqlen + actualSeq] = np.inf
                        golden_gpu_lse_output[:, cu_seqlen: cu_seqlen + actualSeq] = np.inf
                    elif preTokensError != 0:
                        output[cu_seqlen + actualSeq: cu_seqlen  + q_seqlen, :, :] = 0  # 后n行置0
                        golden_gpu_output[cu_seqlen + actualSeq: cu_seqlen + q_seqlen, :, :] = 0
                        golden_lse_output[:, cu_seqlen + actualSeq: cu_seqlen  + q_seqlen] =  np.inf
                        golden_gpu_lse_output[:, cu_seqlen + actualSeq: cu_seqlen + q_seqlen] =  np.inf
            else:
                output[i * max_q_seqlen: i * max_q_seqlen + q_seqlen, :, :] = out
                golden_gpu_output[i * max_q_seqlen: i * max_q_seqlen + q_seqlen, :, :] = out_gpu

                golden_lse_output[:, i * max_q_seqlen: i * max_q_seqlen + q_seqlen] = lse
                golden_gpu_lse_output[:, i * max_q_seqlen: i * max_q_seqlen + q_seqlen] = lse_gpu
                if actualSeq != q_seqlen and sparseMode == 4:
                    if nextTokensError != 0:
                        output[i * max_q_seqlen: i * max_q_seqlen + actualSeq, :, :] = 0
                        golden_gpu_output[i * max_q_seqlen: i * max_q_seqlen + actualSeq, :, :] = 0

                        golden_lse_output[:, i * max_q_seqlen: i * max_q_seqlen + actualSeq] = np.inf
                        golden_gpu_lse_output[:, i * max_q_seqlen: i * max_q_seqlen + actualSeq] = np.inf
                    elif preTokensError != 0:
                        output[i * max_q_seqlen + actualSeq : i * max_q_seqlen + q_seqlen, :, :] = 0
                        golden_gpu_output[i * max_q_seqlen + actualSeq : i * max_q_seqlen + q_seqlen, :, :] = 0

                        golden_lse_output[:, i * max_q_seqlen + actualSeq : i * max_q_seqlen + q_seqlen] = np.inf
                        golden_gpu_lse_output[:, i * max_q_seqlen + actualSeq : i * max_q_seqlen + q_seqlen] = np.inf
            
            cu_seqlen += q_seqlen
            kv_seqlen_now += k_seqlen
    
    def calc_data(self, attention_inputs:AttentionInputs):
        num_tokens = attention_inputs.query.shape[0]
        shape_out = (num_tokens, attention_inputs.auxAttrs.num_heads, attention_inputs.auxAttrs.head_size)
        golden_output = np.zeros(shape_out, dtype=np.float32)
        golden_gpu_output = np.zeros(shape_out, dtype=np.float32)

        lse_shape_out = (attention_inputs.auxAttrs.num_heads, num_tokens)
        golden_lse_output = np.zeros(lse_shape_out, dtype=np.float32)
        golden_gpu_lse_output = np.zeros(lse_shape_out, dtype=np.float32)

        self.ref_single_query_cached_kv_attention(
            attention_inputs,
            golden_output,
            golden_gpu_output,
            golden_lse_output,
            golden_gpu_lse_output
        )

        golden_lse_output = np.transpose(golden_lse_output, (1, 0))
        golden_lse_output = np.expand_dims(golden_lse_output, axis=2)
        golden_gpu_lse_output = np.transpose(golden_gpu_lse_output, (1, 0))
        golden_gpu_lse_output = np.expand_dims(golden_gpu_lse_output, axis=2)

        return golden_output, golden_gpu_output, golden_lse_output, golden_gpu_lse_output
        

def gen_list_from_cumSum(seqlenArray):
    seqlenList = []
    preSeqSum = 0
    for i in range(len(seqlenArray)):
        seqlenList.append(seqlenArray[i] - preSeqSum)
        preSeqSum = seqlenArray[i]
    return seqlenList

def gen_actual_seqlen_list_golden(actualseqlengths, actualseqlengthskv, inputLayout, pagedAttentionFlag):
    qSeqlenList = []
    kvSeqlenList = []
    if inputLayout == 'TND':
        qSeqlenList = gen_list_from_cumSum(actualseqlengths)
        if pagedAttentionFlag:
            kvSeqlenList = list(actualseqlengthskv)
        else:
            kvSeqlenList = gen_list_from_cumSum(actualseqlengthskv)
    else:
        qSeqlenList = list(actualseqlengths)
        kvSeqlenList = list(actualseqlengthskv)
    return qSeqlenList, kvSeqlenList


def create_binary_matrix(qSeqlen, kvSeqlen, preToken, nextToken):
    preToken = kvSeqlen - qSeqlen - preToken
    nextToken = kvSeqlen - qSeqlen + nextToken
    matrix = [[0 for _ in range(kvSeqlen)] for _ in range(qSeqlen)]
    for i in range(qSeqlen):
        for j in range(kvSeqlen):
            is_below_pretoken_line = (-i + j) < preToken
            is_above_nexttoken_line = (-i + j) > nextToken
            if is_below_pretoken_line or is_above_nexttoken_line:
                matrix[i][j] = 1
    
    return np.array(matrix)

def aclnn_op_func_fia_split_fuse_golden(input_data : InputDataset, is_benchmark_task):
    input_data_dtype = input_data.kwargs["query"].dtype

    if input_data_dtype == torch.float16:
         query = input_data.kwargs["query"].numpy()
    elif input_data_dtype == torch.bfloat16:
        query = input_data.kwargs["query"].to(torch.float32).numpy().astype(bfloat16)
    else:
        query = input_data.kwargs["query"].numpy()
        print(f"lwg_高精度_4阶段 \n")

    if input_data_dtype == torch.float16:
        key = input_data.kwargs["key"][0].numpy()
    elif input_data_dtype == torch.bfloat16:
        key = input_data.kwargs["key"][0].to(torch.float32).numpy().astype(bfloat16)
    else:
        key = input_data.kwargs["key"][0].numpy()

    if input_data_dtype == torch.float16:
        value = input_data.kwargs["value"][0].numpy()
    elif input_data_dtype == torch.bfloat16:
        value = input_data.kwargs["value"][0].to(torch.float32).numpy().astype(bfloat16)
    else:
        value = input_data.kwargs["value"][0].numpy()

    # query = input_data.kwargs["query"].numpy() if input_data_dtype == torch.float16 else input_data.kwargs["query"].to(torch.float32).numpy().astype(bfloat16)
    # key = input_data.kwargs["key"][0].numpy() if input_data_dtype == torch.float16 else input_data.kwargs["key"][0].to(torch.float32).numpy().astype(bfloat16)
    # value = input_data.kwargs["value"][0].numpy() if input_data_dtype == torch.float16 else input_data.kwargs["value"][0].to(torch.float32).numpy().astype(bfloat16)
    blockTable = None
    pagedAttentionFlag = False
    if input_data.kwargs["blockTableOptional"] != None:
        blockTable = input_data.kwargs["blockTableOptional"].numpy()
        pagedAttentionFlag = True
    ## gen actual seqlen
    inputLayout = input_data.kwargs["inputLayout"]
    batch = len(input_data.kwargs["actualSeqLengthsOptional"])
    actualseqlengths = [0] * batch
    actualseqlengthsKv = [0] * batch
    for i in range(batch):
        actualseqlengths[i] = input_data.kwargs["actualSeqLengthsOptional"][i]
        actualseqlengthsKv[i] = input_data.kwargs["actualSeqLengthsKvOptional"][i]
    qSeqlenList, kvSeqlenList = gen_actual_seqlen_list_golden(actualseqlengths, actualseqlengthsKv, inputLayout, pagedAttentionFlag)
    maxKvSeqlen = max(kvSeqlenList)
    maxQSeqlen = max(qSeqlenList)
    totalQTokens = sum(qSeqlenList)
    preTokens = input_data.kwargs["preTokens"]
    nextTokens = input_data.kwargs["nextTokens"]
    ## gen mask
    fullMask = None
    pre_mask_factor = -3e38 if input_data_dtype == torch.bfloat16 or input_data_dtype == torch.float32 else -6e4
    if input_data.kwargs["attenMaskOptional"] != None and input_data.kwargs["sparseMode"] == 3:
        maskDtype = dtypeMap[input_data_dtype]
        fullMask = np.zeros(shape=(totalQTokens, maxKvSeqlen)).astype(maskDtype)
        prevQseqlen = 0
        for i in range(len(qSeqlenList)):
            qSeqlen = qSeqlenList[i]
            kSeqlen = kvSeqlenList[i]
            tri = np.ones((qSeqlen, qSeqlen))
            tri = np.triu(tri, 1)
            tri *= pre_mask_factor
            fullMask[prevQseqlen : (prevQseqlen + qSeqlen), kSeqlen - qSeqlen : kSeqlen] = tri
            prevQseqlen += qSeqlen

    if  input_data.kwargs["sparseMode"] == 4:
        maskDtype = dtypeMap[input_data_dtype]
        # fullMask = np.zeros(shape=(totalQTokens, maxKvSeqlen)).astype(np.float16)
        fullMask = np.zeros(shape=(totalQTokens, maxKvSeqlen)).astype(maskDtype)
        prevQseqlen = 0
        for i in range(len(qSeqlenList)):
            qSeqlen = qSeqlenList[i]
            kSeqlen = kvSeqlenList[i]
            # tri = create_complex_mask(qSeqlen, kSeqlen, preTokens, nextTokens)
            tri = create_binary_matrix(qSeqlen, kSeqlen, preTokens, nextTokens)
            # print(f"ljl-preTokens:{preTokens},nextTokens:{nextTokens},qSeqlen:{qSeqlen},kSeqlen:{kSeqlen},maxKvSeqlen:{maxKvSeqlen}")
            print(f"ljl-i:{i},kSeqlen:{kSeqlen},qSeqlen{qSeqlen},prevQseqlen{prevQseqlen},fullMask:{fullMask.shape},maskDtype{maskDtype},{tri.dtype}")
            tri = tri.astype(maskDtype)
            # tri = tri.astype(np.float16)
            tri *= pre_mask_factor
            fullMask[prevQseqlen : (prevQseqlen + qSeqlen), :kSeqlen] = tri
            prevQseqlen += qSeqlen

    learnable_sink = None 
    if input_data.kwargs.get("learnableSinkOptional") is not None:
        print("golden, sink不为空 \n")
        # 格式转换：torch张量 → numpy数组（和query保持一致的dtype规则）
        sink_torch_dtype = input_data.kwargs["learnableSinkOptional"].dtype
        if sink_torch_dtype == torch.float16:
            # 可以报错
            # print(f"[ERROR] learnableSinkOptional dtype is {sink_torch_dtype} (float16), not allowed!")
            # sys.exit(-99)
            print("sink_fp16 \n")
            learnable_sink = input_data.kwargs["learnableSinkOptional"].numpy()
            print(f"sink_fp16: {learnable_sink}")
            learnable_sink = learnable_sink.astype(np.float32)
            print(f"sink_fp32: {learnable_sink}")
        elif sink_torch_dtype == torch.bfloat16:
            print("sink_bf16 \n")
            learnable_sink = input_data.kwargs["learnableSinkOptional"].to(torch.float32).numpy()
            learnable_sink_bf16 = np.array(learnable_sink, dtype=bfloat16)
        # 2. 强制第三头的sink值为BF16近似值（关键：和NPU侧完全一致）
            learnable_sink = learnable_sink_bf16.astype(np.float32)
            print(f"learnable_sink: {learnable_sink} \n")
        else:
            # bf16场景：先转float32再转bfloat16
            # learnable_sink = input_data.kwargs["learnableSinkOptional"].to(torch.float32).numpy()
            # # bf16场景：先转float32再转bfloat16
            print("sink_fp32\n")
            learnable_sink = input_data.kwargs["learnableSinkOptional"].numpy()
            print(f"learnable_sink: {learnable_sink} \n")
    else:
        print("golden, sink为空 \n")
    
    numHeads = input_data.kwargs["numHeads"]
    # if numHeads == 0:
        # exit(-100)
    kvHeads = input_data.kwargs["numKeyValueHeads"]
    headSize = query.shape[2] if inputLayout == 'TND' else 0
    numBlocks = key.shape[0] if pagedAttentionFlag == True else 0
    blockSize = input_data.kwargs["blockSize"]
    print(f"lwg-blockSize-coming:{blockSize} \n")
    maskType = maskTypeMap[input_data.kwargs["sparseMode"]]
    sparseMode = input_data.kwargs["sparseMode"]
    dtype = dtypeMap[input_data_dtype]
    kvOrgMode = 1 if pagedAttentionFlag == True else 0
    layoutMode = 1 if inputLayout == 'TND' else 0
    goldenGpuPrecision = input_data.kwargs["innerPrecise"]
    softmaxLseFlag = input_data.kwargs["softmaxLseFlag"]
    scale = float(input_data.kwargs["scaleValue"])

    if pagedAttentionFlag == True:
        key = key.reshape(key.shape[:-1] + (kvHeads, headSize))
        value = value.reshape(value.shape[:-1] + (kvHeads, headSize))
    # print(key.shape)
    testObj = TestFIAV4SplitFuse()
    auxAttrs = testObj.AuxAttrs(preTokens, nextTokens, numHeads, kvHeads, headSize, numBlocks, blockSize, maskType, dtype, kvOrgMode, layoutMode, maxQSeqlen, maxKvSeqlen, goldenGpuPrecision, scale, sparseMode)
    attentionInputs = testObj.AttentionInputs(query, key, value, blockTable, qSeqlenList, kvSeqlenList, fullMask, learnable_sink, auxAttrs)

    golden_output, golden_gpu_output, golden_lse_output, golden_gpu_lse_output = testObj.calc_data(attentionInputs)
    if golden_output.dtype == "bfloat16":
        print("=================================走入bf16分支",golden_output.dtype)
        golden_output = torch.from_numpy(golden_output.astype(np.float32))
        print(f"golden_output_dtype:{golden_output.dtype} \n")
        golden_gpu_output = torch.from_numpy(golden_gpu_output.astype(np.float32))
        golden_lse_output = torch.from_numpy(golden_lse_output.astype(np.float32))
        golden_gpu_lse_output = torch.from_numpy(golden_gpu_lse_output.astype(np.float32))
    else:
        golden_output = torch.from_numpy(golden_output)
        golden_gpu_output = torch.from_numpy(golden_gpu_output)
        golden_lse_output = torch.from_numpy(golden_lse_output)
        golden_gpu_lse_output = torch.from_numpy(golden_gpu_lse_output)
    # golden_output = torch.from_numpy(golden_output)
    # golden_gpu_output = torch.from_numpy(golden_gpu_output)
    # golden_lse_output = torch.from_numpy(golden_lse_output)
    # golden_gpu_lse_output = torch.from_numpy(golden_gpu_lse_output)
    if not softmaxLseFlag:
        golden_lse_output = torch.tensor([])
        golden_gpu_lse_output = torch.tensor([])

    if not is_benchmark_task:
        # 标杆返回这个
        return golden_gpu_output, golden_gpu_lse_output
    else:
        # 真值返回下面的
        return golden_output, golden_lse_output

@register("executor_fused_infer_attention_score_v4")
class fusedInferAttentionScoreApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(fusedInferAttentionScoreApi, self).__init__(task_result)
    
    def init_by_input_data(self, input_data: InputDataset):
        # np.random.seed(10)
        np.random.seed(self.task_result.case_config.id)
        is_ifa_perf = False
        if is_ifa_perf:
            # 测性能的时 为了获取ifa/pfa的性能需要开启is_ifa_perf=True
            input_data.kwargs["preTokens"]=None
            input_data.kwargs["nextTokens"]=None
            input_data.kwargs["sparseMode"]=3
    
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.name == "perf" or "abnormal" in self.task_result.case_config.name:
            return torch.Tensor([1])

        original_ls_tensor = input_data.kwargs["learnableSinkOptional"]
        print(f"original_ls_tensor:{original_ls_tensor} \n")
        # query = input_data.kwargs["query"]
        # query[1, :, :] = 10
        # original_ls_tensor[0] = 5
        # print(f"original_ls_tensor:{original_ls_tensor} \n")
        # 2. 动态获取原Tensor的长度（关键：不再硬编码6）
        #    - 1维Tensor：shape[0] 就是长度；多维Tensor可根据需求取对应维度
        # ls_length = original_ls_tensor.shape[0]
        # if self.task_result.is_benchmark_task:
        #     ls_tensor = torch.zeros(ls_length, dtype=torch.float32) 
        # else:
        #     ls_tensor = torch.zeros(ls_length, dtype=torch.bfloat16) 
        # ls_tensor.fill_(10)
        # input_data.kwargs["learnableSinkOptional"]=None
        # original_ls_tensor[0] = 20
        # original_ls_tensor[1] = 20
        # original_ls_tensor[2] = 40
        # original_ls_tensor[3] = 40
        # input_data.kwargs["softmaxLseFlag"] = False
        output, output_lse = aclnn_op_func_fia_split_fuse_golden(input_data, self.task_result.is_benchmark_task)
        if input_data.kwargs["softmaxLseFlag"] == True:
            if IS_INF_FLAG:
                print(f"output_golden: {output} \n")
                print(f"output_lse_golden: {output_lse} \n")
            return output, output_lse
        else:
            if IS_INF_FLAG:
                print(f"output_golden: {output} \n")
            return output

@register("executor_aclnn_fused_infer_attention_score_v4")
class aclnnFusedInferAttentionScoreApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        print("lwg_1 \n")
        super(aclnnFusedInferAttentionScoreApi, self).__init__(task_result, backend)
    
    def gen_compressed_triU_mask(self, dim_num, mask_dtype):
        mask_shape_four_dims = (1, 1, 2048, 2048)
        mask_four_dims = torch.zeros(mask_shape_four_dims, dtype = mask_dtype)
        mask_triU = torch.triu(torch.ones(2048, 2048), diagonal=1)
        mask_four_dims[:] = mask_triU
        if dim_num == 2:
            return mask_four_dims[0][0]
        elif dim_num == 3:
            return mask_four_dims[0]
        elif dim_num == 4:
            return mask_four_dims
        else:
            print("invalid dim num, will provide a four dim mask anyway")
            return mask_four_dims
        return mask_four_dims[0][0]
    
    def init_by_input_data(self, input_data: InputDataset):
        # np.random.seed(10)
        np.random.seed(self.task_result.case_config.id)
        # np.random.seed(44)
        if input_data.kwargs["attenMaskOptional"] != None:
            mask_dim_num = len(input_data.kwargs["attenMaskOptional"].shape)
            mask_dtype = input_data.kwargs["attenMaskOptional"].dtype
            input_data.kwargs["attenMaskOptional"] = self.gen_compressed_triU_mask(mask_dim_num, mask_dtype).npu()
        
        original_ls_tensor = input_data.kwargs["learnableSinkOptional"]
        query = input_data.kwargs["query"]
        # query[1, :, :] = 10
        # original_ls_tensor[0] = 5
        print(f"original_ls_tensor:{original_ls_tensor} \n")
        print(f"query: {query} \n")
        # print()
        # 2. 动态获取原Tensor的长度（关键：不再硬编码6）
        #    - 1维Tensor：shape[0] 就是长度；多维Tensor可根据需求取对应维度
        # ls_length = original_ls_tensor.shape[0]   
        # if self.task_result.is_benchmark_task:
        #     ls_tensor = torch.zeros(ls_length, dtype=torch.float32) 
        # else:
        #     ls_tensor = torch.zeros(ls_length, dtype=torch.bfloat16) 
        # ls_tensor.fill_(10)
        # input_data.kwargs["learnableSinkOptional"]=None
        # original_ls_tensor[0] = 20
        # original_ls_tensor[1] = 20
        # original_ls_tensor[2] = 40
        # original_ls_tensor[3] = 40
        # print(f"original_ls_tensor:{original_ls_tensor} \n")
        q_seqLen = input_data.kwargs["actualSeqLengthsOptional"]
        print(f"actualSeqLengthsOptional: {q_seqLen} \n")
        kv_seqLen = input_data.kwargs["actualSeqLengthsKvOptional"]
        print(f"actualSeqLengthsKvOptional: {kv_seqLen} \n")
        # input_data.kwargs["softmaxLseFlag"] = False

        input_args = []  # 算子的入参列表
        input_args, output_packages = super().init_by_input_data(input_data)

        # 将所有type是tensor values是None的输入 改为AclTensor类型的空指针
        for i, (name, kwarg) in enumerate(input_data.kwargs.items()):
            if kwarg is None and self.task_result.case_config.inputs[i].type == "tensor":
                from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
                input_args[i] = TensorPtr()
        
        # input_data.kwargs["learnableSinkOptional"] = None

            
        output_packages = []  # 算子的出参数据包列表
        input_args.pop()
        if input_data.kwargs["softmaxLseFlag"] == True:
            input_args.pop()
            output_packages.append(input_args[-2])
            output_packages.append(input_args[-1])
        else:
            output_packages.append(input_args[-2])
        return input_args, output_packages
    
    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        output = []
        i=0
        for output_pack in output_packages:
            temp_output_pack = self.acl_tensor_to_torch(output_pack).to(dtype=torch.float)
            output.append(temp_output_pack)
            print(f"out_npu_dtype: {temp_output_pack.dtype}")
            if IS_INF_FLAG:
                print(f"npu_temp_output_pack[{i}]: {temp_output_pack} \n")
                i=i+1

        return output