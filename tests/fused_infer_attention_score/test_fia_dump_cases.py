#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate the migrated npu_fused_infer_attention_score against real model dump
files produced by VLLM_ASCEND_DUMP_FIA.

The three target dumps are:
  - fia_layer30_rank0_stateChunkedPrefill.pt
  - fia_layer30_rank0_stateDecodeOnly.pt
  - fia_layer30_rank0_statePrefillNoCache.pt

The original dumps used ``antiquant_mode`` to inject the BlasST threshold:
  sparse_lamda = (antiquant_mode + 100) / 10.0
This script converts that value to the migrated operator's ``sparse_lambda``
parameter and removes ``antiquant_mode`` from the call.

Expected outcome: all three dumps execute successfully.
"""

import argparse
import os
import sys
import traceback

# Point CANN runtime to the custom operator package before NPU init.
_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
_CUSTOM_OPP_PATH = os.path.join(
    _CUR_DIR, "..", "..", "vllm_ascend", "_cann_ops_custom", "vendors", "vllm-ascend"
)
if os.path.exists(_CUSTOM_OPP_PATH):
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = _CUSTOM_OPP_PATH

import torch
import torch_npu  # noqa: F401

from blasst_golden_tnd import BlasstGoldenTND


def _progress(msg: str):
    print(msg)
    sys.stdout.flush()


def antiquant_mode_to_sparse_lambda(antiquant_mode: int) -> float:
    """Source-warehouse mapping: sparse_lamda = (antiquant_mode + 100) / 10.0."""
    return (float(antiquant_mode) + 100.0) / 10.0


def parse_blocknum_from_lse(lse: torch.Tensor, max_cores: int = 24):
    """Parse per-core block sparsity stats from the gLSE buffer.

    In LSE OUT_ONLY mode the kernel writes ``[blockSparseCount, blockCount]``
    into the first two floats of each core's 16-float stride at the head of
    the LSE buffer.  Returns ``(block_sparse_num, block_count, details)``.
    Reference: run_fia_single_op.py::_parse_blocknum_from_lse.

    The A3 part runs the regular kernel on 24 cube cores; slots belonging to
    unused cores contain stale garbage, so the caller must pass the real
    core count (24 for the prefill/regular kernel).
    """
    data = lse.detach().cpu().flatten()
    if len(data) < 2:
        return 0, 0, []
    core_num = min(max_cores, len(data) // 16)
    block_sparse_num = 0
    block_count = 0
    details = []
    non_int_warnings = []
    for core in range(core_num):
        sp_val = data[core * 16].item()
        total_val = data[core * 16 + 1].item()
        sp = int(sp_val)
        total = int(total_val)
        # Warn if parsed counter values are non-integer (garbage / uninitialised data).
        if abs(sp_val - sp) > 1e-6:
            non_int_warnings.append(f"core {core} sparseCount={sp_val:.6f} (non-integer)")
        if abs(total_val - total) > 1e-6:
            non_int_warnings.append(f"core {core} blockCount={total_val:.6f} (non-integer)")
        block_sparse_num += sp
        block_count += total
        details.append({"core": core, "sparse": sp, "total": total,
                        "rate": (sp / total) if total > 0 else 0.0})
    if non_int_warnings:
        print(f"[WARN] parse_blocknum_from_lse: {len(non_int_warnings)} core(s) "
              f"have non-integer counter values (debug region may contain stale data):")
        for w in non_int_warnings:
            print(f"  {w}")
    return block_sparse_num, block_count, details


def reconstruct_golden_inputs(inputs: dict):
    """Reconstruct flat TND tensors + masks for the BlasST golden.

    Returns ``(query, key_tnd, value_tnd, q_cum, kv_cum, masks)`` where
    ``q_cum`` / ``kv_cum`` are cumulative length tensors and ``masks`` is a
    per-batch list (``None`` for batches where the mask does not apply,
    e.g. q_len=1 decode).
    """
    query = inputs["query"].cpu()
    block_table = inputs.get("block_table")
    block_size = int(inputs["block_size"])
    head_dim = query.size(-1)

    q_cum = torch.tensor(inputs["actual_seq_lengths"], dtype=torch.int64)
    kv_raw = [int(x) for x in inputs["actual_seq_lengths_kv"]]

    if isinstance(block_table, torch.Tensor):
        # Paged cache: key/value are (num_blocks, block_size, head_dim) and
        # actual_seq_lengths_kv holds PER-BATCH kv lengths (non-cumulative).
        cache_k = inputs["key"].cpu()
        cache_v = inputs["value"].cpu()
        k_chunks, v_chunks = [], []
        for b, kv_len in enumerate(kv_raw):
            n_blocks = (kv_len + block_size - 1) // block_size
            blk_idx = block_table[b, :n_blocks].long().cpu()
            k_b = cache_k[blk_idx].reshape(-1, head_dim)[:kv_len].unsqueeze(1)
            v_b = cache_v[blk_idx].reshape(-1, head_dim)[:kv_len].unsqueeze(1)
            k_chunks.append(k_b)
            v_chunks.append(v_b)
        key_tnd = torch.cat(k_chunks, dim=0)
        value_tnd = torch.cat(v_chunks, dim=0)
        kv_cum = torch.cumsum(torch.tensor(kv_raw, dtype=torch.int64), dim=0)
    else:
        key_tnd = inputs["key"].cpu()
        value_tnd = inputs["value"].cpu()
        kv_cum = torch.tensor(kv_raw, dtype=torch.int64)

    # Per-batch masks: only batches whose q_len matches the mask rows use it.
    atten_mask = inputs.get("atten_mask")
    masks = []
    q_start = 0
    for b in range(q_cum.numel()):
        q_len = int(q_cum[b].item()) - q_start
        q_start = int(q_cum[b].item())
        if isinstance(atten_mask, torch.Tensor) and q_len == atten_mask.size(0):
            masks.append(atten_mask.cpu())
        else:
            masks.append(None)

    return query, key_tnd, value_tnd, q_cum, kv_cum, masks


def run_golden_blasst(inputs: dict, sparse_lambda: float, golden_block_size: int = 32,
                      golden_mode: str = "block32"):
    """Run the BlasST golden on the dump inputs with the given sparse_lambda.

    ``golden_mode``:
      - ``block32``: legacy 32-wide block, per-(token, head) skip granularity.
      - ``kernel``:  kernel-exact granularity (qSBlock=128 x head x KV stack
        of 512, two 64-row subBlocks x 16-row rowLoops).
    """
    query, key_tnd, value_tnd, q_cum, kv_cum, masks = \
        reconstruct_golden_inputs(inputs)
    golden = BlasstGoldenTND(
        num_heads=int(inputs["num_heads"]),
        num_key_value_heads=int(inputs["num_key_value_heads"]),
        head_dim=query.size(-1),
        scale=float(inputs["scale"]),
        block_size=golden_block_size,
        sparse_lamda=sparse_lambda,
    )
    if golden_mode == "kernel":
        out_ref, lse_ref, info = golden.forward_blasst_kernel(
            query, key_tnd, value_tnd, q_cum, kv_cum, masks=masks)
    else:
        out_ref, lse_ref, info = golden.forward_blasst(
            query, key_tnd, value_tnd, q_cum, kv_cum, masks=masks)
    return out_ref, info


def load_dump(path: str):
    data = torch.load(path, map_location="cpu", weights_only=False)
    inputs = data["inputs"]
    output = data.get("output")
    meta = {k: data[k] for k in ("layer_idx", "tp_rank", "attn_state")
            if k in data}
    return meta, inputs, output


def build_call_kwargs(inputs: dict, device: torch.device,
                      sparse_lambda_override: float = None):
    """Build kwargs for ``torch.ops._C_ascend.npu_fused_infer_attention_score``."""
    kwargs = {}

    # Tensors: move to NPU unless None.
    tensor_keys = [
        "query", "key", "value", "atten_mask", "block_table",
        "actual_seq_lengths", "actual_seq_lengths_kv",
    ]
    for k in tensor_keys:
        v = inputs.get(k)
        if isinstance(v, torch.Tensor):
            kwargs[k] = v.to(device)
        elif isinstance(v, (list, tuple)) and k.startswith("actual_seq_lengths"):
            kwargs[k] = torch.tensor(v, dtype=torch.int64, device=device)
        elif v is not None:
            kwargs[k] = v

    # Scalar / string parameters.
    for k in ("num_heads", "num_key_value_heads", "block_size",
              "sparse_mode", "inner_precise", "input_layout"):
        if k in inputs and inputs[k] is not None:
            kwargs[k] = inputs[k]

    # Scale is a Python float or scalar tensor.
    scale = inputs.get("scale")
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    kwargs["scale"] = float(scale)

    # Convert antiquant_mode to sparse_lambda and drop antiquant_mode.
    antiquant_mode = inputs.get("antiquant_mode")
    if isinstance(antiquant_mode, torch.Tensor):
        antiquant_mode = int(antiquant_mode.item())
    if sparse_lambda_override is not None:
        sparse_lambda = float(sparse_lambda_override)
    else:
        sparse_lambda = antiquant_mode_to_sparse_lambda(antiquant_mode) \
            if antiquant_mode is not None else -99.0
    kwargs["sparse_lambda"] = sparse_lambda
    kwargs["antiquant_mode"] = 0  # migrated adapter requires 0

    # Default FIA parameters.
    kwargs.setdefault("pse_shift", None)
    kwargs.setdefault("pre_tokens", 2147483647)
    kwargs.setdefault("next_tokens", 2147483647)
    kwargs.setdefault("softmax_lse_flag", True)

    return kwargs, sparse_lambda


def run_single_case(dump_path: str, device: torch.device, no_compare: bool,
                    sparse_lambda_override: float = None,
                    ref_output: torch.Tensor = None, ref_label: str = "",
                    golden: bool = True, golden_mode: str = "block32"):
    _progress("=" * 80)
    tag = f" (lambda={sparse_lambda_override})" if sparse_lambda_override is not None else ""
    _progress(f"Processing: {os.path.basename(dump_path)}{tag}")
    _progress("=" * 80)

    meta, inputs, dumped_output = load_dump(dump_path)
    _progress(f"Meta: {meta}")

    kwargs, sparse_lambda = build_call_kwargs(inputs, device, sparse_lambda_override)
    if sparse_lambda_override is not None:
        _progress(f"Overridden sparse_lambda={sparse_lambda:.2f} "
                  f"(dump antiquant_mode={inputs.get('antiquant_mode')})")
    else:
        _progress(f"Converted sparse_lambda={sparse_lambda:.2f} from antiquant_mode={inputs.get('antiquant_mode')}")

    # Print key shapes for quick inspection.
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            _progress(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
        else:
            _progress(f"  {k}: {v}")

    result = {
        "file": os.path.basename(dump_path),
        "meta": meta,
        "sparse_lambda": sparse_lambda,
        "lambda_overridden": sparse_lambda_override is not None,
        "status": "PENDING",
        "error": None,
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "ref_max_abs_diff": None,
        "ref_mean_abs_diff": None,
        "ref_label": ref_label,
        "golden_max_abs_diff": None,
        "golden_mean_abs_diff": None,
        "golden_sparsity": None,
        "npu_block_sparse_num": None,
        "npu_block_count": None,
        "npu_sparse_rate": None,
        "output_shape": None,
        "lse_shape": None,
        "out": None,
    }

    try:
        with torch.npu.device(device):
            out, lse = torch.ops._C_ascend.npu_fused_infer_attention_score(
                kwargs["query"],
                kwargs["key"],
                kwargs["value"],
                pse_shift=kwargs.get("pse_shift"),
                atten_mask=kwargs.get("atten_mask"),
                actual_seq_lengths=kwargs["actual_seq_lengths"],
                actual_seq_lengths_kv=kwargs["actual_seq_lengths_kv"],
                blocktable=kwargs.get("block_table"),
                num_heads=kwargs["num_heads"],
                scale=kwargs["scale"],
                pre_tokens=kwargs["pre_tokens"],
                next_tokens=kwargs["next_tokens"],
                input_layout=kwargs["input_layout"],
                num_key_value_heads=kwargs["num_key_value_heads"],
                sparse_mode=kwargs["sparse_mode"],
                inner_precise=kwargs.get("inner_precise", 0),
                block_size=kwargs["block_size"],
                antiquant_mode=kwargs["antiquant_mode"],
                sparse_lambda=kwargs["sparse_lambda"],
                softmax_lse_flag=kwargs["softmax_lse_flag"],
            )
            torch.npu.synchronize()

        result["status"] = "OK"
        result["output_shape"] = tuple(out.shape)
        result["lse_shape"] = tuple(lse.shape)
        _progress(f"  Output shape: {result['output_shape']} dtype={out.dtype}")
        _progress(f"  LSE shape:    {result['lse_shape']} dtype={lse.dtype}")
        _progress(f"  LSE stats: min={lse.min().item():.6f} max={lse.max().item():.6f} "
                  f"mean={lse.mean().item():.6f}")

        # Parse per-core block sparsity counters embedded in the gLSE head.
        sp_num, blk_cnt, core_details = parse_blocknum_from_lse(lse)
        result["npu_block_sparse_num"] = sp_num
        result["npu_block_count"] = blk_cnt
        result["npu_sparse_rate"] = (sp_num / blk_cnt) if blk_cnt > 0 else 0.0
        result["npu_core_details"] = core_details
        _progress(f"  NPU block sparse (from gLSE): sparse={sp_num} total={blk_cnt} "
                  f"rate={result['npu_sparse_rate']:.2%}")
        for d in core_details:
            _progress(f"    core {d['core']:>2}: sparse={d['sparse']:>6} total={d['total']:>6} "
                      f"rate={d['rate']:.2%}")

        if not no_compare and dumped_output is not None:
            expected = dumped_output.to(device)
            if out.shape != expected.shape:
                result["error"] = f"shape mismatch: expected {tuple(expected.shape)}, got {tuple(out.shape)}"
                result["status"] = "SHAPE_MISMATCH"
            else:
                out_cpu = out.detach().cpu().float()
                expected_cpu = expected.detach().cpu().float()
                diff = (out_cpu - expected_cpu).abs()
                result["max_abs_diff"] = diff.max().item()
                result["mean_abs_diff"] = diff.mean().item()
                _progress(f"  vs dumped output: max_abs_diff={result['max_abs_diff']:.6e} "
                          f"mean_abs_diff={result['mean_abs_diff']:.6e}")

        # Extra comparison against a reference NPU run (e.g. original lambda).
        if ref_output is not None and result["status"] == "OK" \
                and out.shape == ref_output.shape:
            ref_cpu = ref_output.detach().cpu().float()
            ref_diff = (out.detach().cpu().float() - ref_cpu).abs()
            result["ref_max_abs_diff"] = ref_diff.max().item()
            result["ref_mean_abs_diff"] = ref_diff.mean().item()
            _progress(f"  vs {ref_label}: max_abs_diff={result['ref_max_abs_diff']:.6e} "
                      f"mean_abs_diff={result['ref_mean_abs_diff']:.6e}")

        # Golden comparison: BlasST golden on the same inputs and lambda.
        if golden and result["status"] == "OK":
            try:
                _progress(f"  Running BlasST golden ({golden_mode}) on dump inputs "
                          f"(lambda={sparse_lambda:.2f}) ...")
                out_golden, g_info = run_golden_blasst(inputs, sparse_lambda,
                                                       golden_mode=golden_mode)
                g_diff = (out.detach().cpu().float() - out_golden.float()).abs()
                result["golden_max_abs_diff"] = g_diff.max().item()
                result["golden_mean_abs_diff"] = g_diff.mean().item()
                result["golden_sparsity"] = g_info["sparsity"]
                _progress(f"  vs golden(BlasST): max_abs_diff={result['golden_max_abs_diff']:.6e} "
                          f"mean_abs_diff={result['golden_mean_abs_diff']:.6e} "
                          f"golden_sparsity={g_info['sparsity']:.2%} "
                          f"({g_info['skipped_blocks']}/{g_info['total_blocks']} blocks)")
                if result["npu_block_count"]:
                    _progress(f"  sparsity compare: NPU={result['npu_sparse_rate']:.2%} "
                              f"vs golden={g_info['sparsity']:.2%}")
            except Exception as ge:
                _progress(f"  [WARN] golden comparison failed: "
                          f"{type(ge).__name__}: {ge}")
                traceback.print_exc()

        # Keep the output for cross-lambda comparison by the caller.
        result["out"] = out.detach().cpu()

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
        _progress(f"  [ERROR] {result['error']}")
        traceback.print_exc()

    _progress("")
    return result


def print_summary(results):
    _progress("=" * 80)
    _progress("Dump validation summary")
    _progress("=" * 80)
    for r in results:
        file_name = r["file"]
        state = r["meta"].get("attn_state", "Unknown")
        status = r["status"]
        lam = r["sparse_lambda"]
        max_diff = r.get("max_abs_diff")
        mean_diff = r.get("mean_abs_diff")
        max_diff_str = f"{max_diff:.6e}" if max_diff is not None else "N/A"
        mean_diff_str = f"{mean_diff:.6e}" if mean_diff is not None else "N/A"
        _progress(f"{file_name:<55} state={state:<18} lambda={lam:<7.2f} status={status:<8} "
                  f"max_diff={max_diff_str:<14} mean_diff={mean_diff_str:<14}")
        if r.get("npu_block_count"):
            _progress(f"    NPU block sparse: {r['npu_block_sparse_num']}/{r['npu_block_count']} "
                      f"rate={r['npu_sparse_rate']:.2%}")
        ref_max = r.get("ref_max_abs_diff")
        ref_mean = r.get("ref_mean_abs_diff")
        if ref_max is not None:
            _progress(f"    vs {r['ref_label']}: max_diff={ref_max:.6e} mean_diff={ref_mean:.6e}")
        g_max = r.get("golden_max_abs_diff")
        g_mean = r.get("golden_mean_abs_diff")
        if g_max is not None:
            _progress(f"    vs golden(BlasST): max_diff={g_max:.6e} mean_diff={g_mean:.6e} "
                      f"golden_sparsity={r['golden_sparsity']:.2%}")
        if r.get("error"):
            _progress(f"  error: {r['error']}")
    _progress("=" * 80)

    # Enforce user expectations: the original (antiquant_mode-derived) lambda
    # runs must all succeed.  Overridden-lambda comparison runs are
    # informational only.
    expectations = {
        "fia_layer30_rank0_statePrefillNoCache.pt": "OK",
        "fia_layer30_rank0_stateChunkedPrefill.pt": "OK",
        "fia_layer30_rank0_stateDecodeOnly.pt": "OK",
    }
    failed = []
    for r in results:
        if r.get("lambda_overridden"):
            continue
        expected = expectations.get(r["file"])
        if expected and r["status"] != expected:
            failed.append((r["file"], expected, r["status"]))

    if failed:
        _progress("ASSERTION FAILED: expected cases did not succeed:")
        for f, exp, got in failed:
            _progress(f"  {f}: expected {exp}, got {got}")
        sys.exit(1)
    else:
        _progress("All dump cases passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Validate migrated FIA against real model dumps")
    parser.add_argument("--device", default="npu:0", help="NPU device (default: npu:0)")
    parser.add_argument("--no-compare", action="store_true",
                        help="Do not compare with dumped output")
    parser.add_argument("--extra-lambdas", type=float, nargs="*", default=[-3.0],
                        help="Extra sparse_lambda values to re-run each dump with "
                             "for precision comparison (default: -3.0)")
    parser.add_argument("--no-golden", action="store_true",
                        help="Skip the BlasST golden comparison on dump inputs")
    parser.add_argument("--golden-mode", choices=["block32", "kernel"],
                        default="block32",
                        help="Golden skip granularity: block32 (legacy 32-wide "
                             "per-token) or kernel (qSBlock=128 x KV stack=512, "
                             "matching the NPU kernel)")
    parser.add_argument("--dump-filter", type=str, default="",
                        help="Only run dumps whose filename contains this substring")
    args = parser.parse_args()

    from vllm_ascend import platform
    platform.NPUPlatform.import_kernels()
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    device = torch.device(args.device)
    torch.npu.set_device(device)

    dump_root = "/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps"
    dump_files = [
        os.path.join(dump_root, "fia_layer30_rank0_stateChunkedPrefill.pt"),
        os.path.join(dump_root, "fia_layer30_rank0_stateDecodeOnly.pt"),
        os.path.join(dump_root, "fia_layer30_rank0_statePrefillNoCache.pt"),
    ]

    results = []
    for dump_path in dump_files:
        if args.dump_filter and args.dump_filter not in os.path.basename(dump_path):
            continue
        if not os.path.exists(dump_path):
            _progress(f"[SKIP] Dump file not found: {dump_path}")
            continue
        base_result = run_single_case(dump_path, device, args.no_compare,
                                      golden=not args.no_golden,
                                      golden_mode=args.golden_mode)
        results.append(base_result)
        # Re-run with extra sparse_lambda thresholds (e.g. -3.0) and compare
        # against the original-lambda NPU output as the precision baseline.
        for lam in args.extra_lambdas:
            if base_result["status"] != "OK" or base_result["out"] is None:
                _progress(f"[SKIP] extra lambda={lam} for {os.path.basename(dump_path)}: "
                          f"base run status={base_result['status']}")
                continue
            results.append(run_single_case(
                dump_path, device, args.no_compare,
                sparse_lambda_override=lam,
                ref_output=base_result["out"],
                ref_label=f"NPU lambda={base_result['sparse_lambda']:.2f}",
                golden=not args.no_golden,
                golden_mode=args.golden_mode))

    print_summary(results)


if __name__ == "__main__":
    main()
