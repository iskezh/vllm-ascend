# npu_fused_infer_attention_score

`npu_fused_infer_attention_score` is a custom aclnn operator migrated from
`ops-transformer/fused_infer_attention_score` (BlasST SplitFuse branch). It
fuses the QK^T matmul, online softmax, PV matmul and output rescale into a
single NPU kernel for Ascend910B3 (`ascend910_93`).

## Where it lives

- Host (tiling / op_def / proto): `csrc/fused_infer_attention_score/op_host/`
- Kernel: `csrc/fused_infer_attention_score/op_kernel/`
- PyTorch adapter: `csrc/fused_infer_attention_score/fused_infer_attention_score_torch_adpt.h`
- Torch binding: `csrc/torch_binding.cpp`
- Unit test: `tests/fused_infer_attention_score/test_fused_infer_attention_score.py`

## Supported scenarios

| Item | Support |
|---|---|
| Layout | `TND` only (`query`/`key`/`value` are 3-D: `[total_tokens, num_heads, head_dim]`) |
| Data type | `torch.float16`, `torch.bfloat16` |
| Head dims | 64 (the tested value; other aligned head dims may work) |
| Batch | Variable-length batch via cumulative `actual_seq_lengths` / `actual_seq_lengths_kv` |
| Mask | No mask (`sparse_mode=0`). Causal mask tiling keys exist but require an external compressed `atten_mask` tensor; passing `sparse_mode=3` without a mask is not verified. |
| Paged cache | Decoding tiling keys are registered (`q_seqlen==1`, no mask, no LSE), but the paged-cache path is not exercised by the current test. |
| LSE output | Supported via `softmax_lse_flag=True` |
| BlasST sparsity | Supported via the dedicated `sparse_lambda` parameter |

## What is NOT supported in this migration

- `pse_shift`
- `atten_mask=None` together with `sparse_mode=3` (causal mask without a real mask tensor)
- `inner_precise != 0`
- `antiquant_mode != 0` (only `0` is accepted; the original `antiquant_mode` meaning is preserved)
- Non-TND layouts (`BSND`, `BNSD`, etc.)

## API

```python
out, lse = torch.ops._C_ascend.npu_fused_infer_attention_score(
    query, key, value,
    pse_shift=None,
    atten_mask=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    blocktable=None,
    num_heads=1,
    scale=1.0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    input_layout='TND',
    num_key_value_heads=0,
    sparse_mode=0,
    inner_precise=0,
    block_size=0,
    antiquant_mode=0,
    sparse_lambda=-99.0,
    softmax_lse_flag=False,
)
```

## Parameter description

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `Tensor` | Yes | - | `(total_q_tokens, num_heads, head_dim)`, TND layout |
| `key` | `Tensor` | Yes | - | `(total_kv_tokens, num_heads, head_dim)` or `(num_kv_heads, ...)` when `num_key_value_heads>0` |
| `value` | `Tensor` | Yes | - | Same shape convention as `key` |
| `pse_shift` | `Tensor` | No | `None` | Not supported in this migration |
| `atten_mask` | `Tensor` | No | `None` | Optional compressed mask; required for causal mask variants |
| `actual_seq_lengths` | `Tensor` | Yes | - | Cumulative Q sequence lengths, `int64`, shape `(batch,)` |
| `actual_seq_lengths_kv` | `Tensor` | Yes | - | Cumulative KV sequence lengths, `int64`, shape `(batch,)` |
| `blocktable` | `Tensor` | No | `None` | Paged-cache block table (`int32`) for decoding |
| `num_heads` | `int` | Yes | `1` | Number of query heads |
| `scale` | `float` | Yes | `1.0` | Attention score scaling factor |
| `pre_tokens` | `int` | No | `INT_MAX` | Sliding-window pre-tokens |
| `next_tokens` | `int` | No | `INT_MAX` | Sliding-window next-tokens |
| `input_layout` | `str` | No | `'TND'` | Only `'TND'` is accepted |
| `num_key_value_heads` | `int` | No | `0` | `0` means equal to `num_heads`; otherwise GQA/MQA head count |
| `sparse_mode` | `int` | No | `0` | `0` = no mask; `3` = compressed causal mask (needs mask tensor) |
| `inner_precise` | `int` | No | `0` | Only `0` is supported |
| `block_size` | `int` | No | `0` | Paged-cache block size; used only for paged path |
| `antiquant_mode` | `int` | No | `0` | Original op attribute; must be `0` in this migration |
| `sparse_lambda` | `float` | No | `-99.0` | **BlasST threshold**. Values `< 0` enable dynamic sparse attention; `-99.0` disables it |
| `softmax_lse_flag` | `bool` | No | `False` | If `True`, also returns `softmax_lse` |

## Outputs

| Output | Shape | Dtype | Description |
|---|---|---|---|
| `attention_out` | Same as `query` | Same as `query` | Attention result |
| `softmax_lse` | `(total_q_tokens, num_heads, 1)` | `float32` | Log-sum-exp of the attention scores (valid when `softmax_lse_flag=True`) |

## BlasST sparsity

`sparse_lambda` is a dedicated parameter for the BlasST sparse attention path.
It replaces the previous hack that hijacked `antiquant_mode`.

- `sparse_lambda = -99.0` (default): BlasST disabled, dense attention.
- `sparse_lambda < 0`: enable dynamic sparsity. The kernel skips KV blocks whose
  max score is below `sparse_lambda`. Common value: `-3.0`.

Example:

```python
out, lse = torch.ops._C_ascend.npu_fused_infer_attention_score(
    query, key, value,
    actual_seq_lengths=actual_seq_lengths,
    actual_seq_lengths_kv=actual_seq_lengths_kv,
    num_heads=4,
    scale=1.0 / math.sqrt(head_dim),
    input_layout="TND",
    sparse_lambda=-3.0,
    softmax_lse_flag=True,
)
```

## Build and test

Build the custom opp package:

```bash
cd vllm-ascend/csrc
bash build.sh -n "fused_infer_attention_score" -c "ascend910_93"
```

Install:

```bash
./build/CANN-custom_ops--linux.aarch64.run \
    --quiet --install-path=/path/to/vllm_ascend/_cann_ops_custom
source /path/to/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash
```

Rebuild the PyTorch extension:

```bash
cd vllm-ascend
MAX_JOBS=8 python setup.py build_ext --inplace
```

Run the unit test:

```bash
python -X faulthandler tests/fused_infer_attention_score/test_fused_infer_attention_score.py
```

## Test coverage

The current unit test validates:

- FP16 dense baseline (`sparse_lambda=-99.0`)
- BF16 dense baseline (`sparse_lambda=-99.0`)
- FP16 BlasST path (`sparse_lambda=-3.0`)
- BF16 BlasST path (`sparse_lambda=-3.0`)

Typical accuracy against a float32 PyTorch reference:

| Dtype | Max abs output diff | LSE diff |
|---|---|---|
| FP16 | ~7.5e-4 | ~0 |
| BF16 | ~6.3e-3 | ~0 |

## Known issues from code review

The following findings come from the `ascendc-code-review` analysis of the
original `ops-transformer/attention/fused_infer_attention_score` source. They
apply to the code that was migrated into this directory and should be addressed
before the operator is considered production-ready.

Overall review result: **FAIL (50/100)**.

### HIGH (must fix)

| ID | Rule | Problem | Key locations |
|---|---|---|---|
| H1 | API-1 | Production code uses `GlobalTensor::GetValue/SetValue`, which performs element-wise GM access and is prohibited in release builds. | `op_kernel/flash_attention_regular.h:251,266-274,329-358`<br>`op_kernel/flash_attention_regular_decode.h:209-243,507-511,526,820-828`<br>`op_kernel/attn_infra/epilogue/block/CombineScale.hpp:119`<br>`op_kernel/attn_infra/gemm/block/block_mmad_pv.hpp:150,226-227`<br>`op_kernel/attn_infra/gemm/block/block_mmad_pv_decode.hpp:144,191-192`<br>`op_kernel/attn_infra/gemm/block/block_mmad_qk.hpp:179`<br>`op_kernel/attn_infra/gemm/block/block_mmad_qk_decode.hpp:160`<br>`op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp:1414` |
| H2 | cpp-secure 2.3 / topk 7 | `GetValueD` divides by `numKeyValueHeads` without checking for zero. The IR default for `num_key_value_heads` is `0`, so omitting the attribute causes division by zero. | `op_host/fused_infer_attention_score_infershape.cpp:180,197,352` |
| H3 | topk 8 / cpp-secure 2.2 | GM offsets and large-shape calculations use `uint32_t`. Products such as `batch * heads * seq * head_dim` can wrap around `UINT32_MAX` and access the wrong GM address. | `op_kernel/attn_infra/epilogue/block/CombineScale.hpp:121,125,129`<br>`op_kernel/flash_attention_regular.h:533,543-544` |

### MEDIUM (should confirm / fix)

| ID | Rule | Problem | Key locations |
|---|---|---|---|
| M1 | API-9 | `float -> half` output cast uses `CAST_NONE` (truncation) instead of `CAST_ROUND`, which may hurt accuracy. | `op_kernel/attn_infra/epilogue/block/CombineScale.hpp:233-237`<br>`op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o.hpp:387-398,609-621`<br>`op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp:1161-1177` |
| M2 | PREC-3 | Online-softmax reduction accumulates in `half` (`WholeReduceSum<half>`, `WholeReduceMax<half>`). Long sequences or extreme distributions may accumulate FP16 error. | `op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax_low_prec.hpp:213,226,261,275,319` |
| M3 | API-10 | `DataCopyParams` and `DataCopyExtParams` have different `blockLen` units (32-byte blocks vs bytes). A `DataCopyParams(1, 1 * sizeof(uint8_t), 0, 0)` call may move 32 bytes instead of 1. | `op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp:202-205` |
| M4 | API-12 | `CrossCoreSetFlag/WaitFlag` pairing is split across files/branches. Static review cannot prove 1:1 pairing on every path, and early `return` may leave a peer core waiting forever. | `flash_attention_regular.h` (`qkReady`, `softmaxReady`) vs `block_epilogue_online_softmax.hpp` |
| M5 | topk 5 | Host code reads Int-type IR attributes into `uint32_t`/`int32_t`, which can truncate or mismatch the IR type (`int64_t`). | `op_host/fallback_fused_infer_attention_score.cpp:213-227`<br>`op_host/fused_infer_attention_score_tiling.cpp:326,332,337` |
| M6 | topk 1 / cpp-secure 3.5 | `GetAttrPointer` results are dereferenced without per-pointer null checks after only verifying `attrs != nullptr`. | `op_host/fused_infer_attention_score_tiling.cpp:324-337`<br>`op_host/fallback_fused_infer_attention_score.cpp:213-227` |

### LOW (nice to fix / document)

| ID | Rule | Problem | Key locations |
|---|---|---|---|
| L1 | API-8 | `repeatTimes` is computed from runtime tile sizes and cast to `uint8_t/uint16_t` without an explicit `<= 255` guard. | `op_kernel/attn_infra/gemm/tile_common/copy_l1_to_l0a.hpp`<br>`op_kernel/attn_infra/gemm/tile_common/copy_l1_to_l0b.hpp`<br>`op_kernel/attn_infra/epilogue/tile_common/tile_broadcast_*.hpp` |
| L2 | PERF-3 | No standard TQue double buffer (`InitBuffer(..., 2, ...)`) is used. Overlap relies on event/ping-pong flags instead. | `op_kernel/attn_infra/arch/local_tensor_buffer.hpp` |
| L3 | PERF-5 | Many small `DataCopy`/`DataCopyPad` calls (LSE rows, sp flags, tails) move far less than the recommended 16 KB per transfer. | Scattered across kernel templates |

### Recommended fix order

1. **Immediately**: H1 (GlobalTensor element access), H2 (division by zero), H3 (32-bit GM offset wraparound).
2. **This iteration**: M1 (Cast round mode), M3 (DataCopy params unit), M4 (CrossCore flag pairing), M5 (attribute types), M6 (null checks).
3. **Later**: M2 (FP32 softmax accumulator), L1 (repeatTimes bounds), L2/L3 (performance/CMake hardening).

## Notes

- The operator relies on `ASCEND_CUSTOM_OPP_PATH` pointing to the installed
  `vllm_ascend/_cann_ops_custom/vendors/vllm-ascend` directory.
- Make sure the PyTorch extension is rebuilt after any change to the custom opp
  package, otherwise ABI mismatches can cause host-side segfaults.
- Causal mask support requires constructing the compressed `atten_mask` tensor
  expected by the kernel; it is not covered by the current minimal migration.
