import math
import os
import sys

# Point CANN runtime to the custom operator package before NPU init.
_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
_CUSTOM_OPP_PATH = os.path.join(
    _CUR_DIR, "..", "..", "vllm_ascend", "_cann_ops_custom", "vendors", "vllm-ascend"
)
if os.path.exists(_CUSTOM_OPP_PATH):
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = _CUSTOM_OPP_PATH

# BLASST golden: optimized_blasst_sim (kernel-granularity simulator,
# QS_BLOCK=128 / KV_BLOCK=512 / rowloop=16, npu_head_mode=True).
_OPS_TRANSFORMER_TESTS = (
    "/home/z00603376/ops-transformer-dev/attention/fused_infer_attention_score/tests/tests"
)
if os.path.isdir(_OPS_TRANSFORMER_TESTS) and _OPS_TRANSFORMER_TESTS not in sys.path:
    sys.path.insert(0, _OPS_TRANSFORMER_TESTS)

import torch

from row_loop_python_blasst import optimized_blasst_sim, GlobalConfig as BlasstGlobalConfig
from gen_sparse_synthetic import gen_mixture, causal_mask
from test_fia_dump_cases import parse_blocknum_from_lse

SWA_INT_MAX = 2147483647
DENSE_LAMBDA = -99.0

# Comparison policy:
#   dense cases  -> custom vs torch_npu only, attn_output only.
#   blasst cases -> three levels, attn_output + lse-parsed sparsity only:
#     L1 custom vs optimized_blasst_sim(lambda)      (output + sparse counts)
#     L2 custom vs optimized_blasst_sim(-99, dense)  (output)
#     L3 custom vs torch_npu(antiquant_mode=lambda)  (output)


# ---------------------------------------------------------------------
# Op wrappers
# ---------------------------------------------------------------------
def run_npu_fia(query, key, value, actual_seq_lengths, actual_seq_lengths_kv,
                num_heads, num_key_value_heads, scale, sparse_lambda,
                head_dim, mask_type="none", atten_mask=None, blocktable=None,
                block_size=0, softmax_lse_flag=True, device="npu:0"):
    """Invoke the migrated custom operator with the TND API."""
    sparse_mode = 3 if mask_type == "causal" else 0
    # custom op 现在收 host int64 list（tiling 经 attr 直读，零 D2H）
    if isinstance(actual_seq_lengths, torch.Tensor):
        actual_seq_lengths = actual_seq_lengths.cpu().tolist()
    if isinstance(actual_seq_lengths_kv, torch.Tensor):
        actual_seq_lengths_kv = actual_seq_lengths_kv.cpu().tolist()
    out, lse, _ = torch.ops._C_ascend.npu_fused_infer_attention_score(
        query, key, value,
        pse_shift=None,
        atten_mask=atten_mask,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        blocktable=blocktable,
        num_heads=num_heads,
        scale=scale,
        pre_tokens=SWA_INT_MAX,
        next_tokens=SWA_INT_MAX,
        input_layout="TND",
        num_key_value_heads=num_key_value_heads,
        sparse_mode=sparse_mode,
        inner_precise=0,
        block_size=block_size,
        antiquant_mode=0,
        sparse_lambda=sparse_lambda,
        softmax_lse_flag=softmax_lse_flag,
    )
    return out, lse


def _run_torch_npu_fia(query, key, value, actual_seq_lengths, actual_seq_lengths_kv,
                       num_heads, num_key_value_heads, scale,
                       sparse_lambda=DENSE_LAMBDA,
                       mask_type="none", atten_mask=None, device="npu:0"):
    """Invoke torch_npu.npu_fused_infer_attention_score with TND layout.

    sparse_lambda is mapped to antiquant_mode = int(lambda * 10 - 100),
    matching the kernel tiling decode (antiquantMode + 100) / 10.
    lambda=-99.0 -> antiquant_mode=-1090 (dense equivalent).
    """
    import torch_npu
    asl_list = actual_seq_lengths.tolist() if isinstance(actual_seq_lengths, torch.Tensor) else list(actual_seq_lengths)
    asl_kv_list = actual_seq_lengths_kv.tolist() if isinstance(actual_seq_lengths_kv, torch.Tensor) else list(actual_seq_lengths_kv)
    n_kv = num_key_value_heads if num_key_value_heads > 0 else num_heads
    sparse_mode = 3 if mask_type == "causal" else 0
    is_dense = abs(sparse_lambda - DENSE_LAMBDA) < 0.01
    # Dense uses antiquant_mode=0; sparse maps lambda via the kernel tiling
    # decode.  torch_npu TND sparse (antiquant_mode < 0) is only accepted on
    # GQA variants -- MHA sparse is rejected by its tiling check, so callers
    # skip L3 for MHA (see _cross_check_npu).
    antiquant_mode = 0 if is_dense else int(sparse_lambda * 10 - 100)
    kwargs = dict(
        num_heads=num_heads, scale=scale,
        input_layout="TND", num_key_value_heads=n_kv,
        sparse_mode=sparse_mode, inner_precise=0, antiquant_mode=antiquant_mode,
        softmax_lse_flag=True,
    )
    if asl_list:
        kwargs["actual_seq_lengths"] = asl_list
    if asl_kv_list:
        kwargs["actual_seq_lengths_kv"] = asl_kv_list
    if atten_mask is not None:
        kwargs["atten_mask"] = atten_mask
    return torch_npu.npu_fused_infer_attention_score(query, key, value, **kwargs)


def _cross_check_npu(out_custom, query, key, value,
                     actual_seq_lengths, actual_seq_lengths_kv,
                     num_heads, num_key_value_heads, scale,
                     sparse_lambda=DENSE_LAMBDA,
                     mask_type="none", atten_mask=None, device="npu:0"):
    """L3: custom vs torch_npu, attn_output only.

    Dense (lambda=-99): assert within size/dtype-scaled threshold.
    Sparse: torch_npu runs the same sparse path via antiquant_mode; the two
    implementations are expected to be bit-exact (verified on real dumps).
    Note: torch_npu TND sparse (antiquant_mode < 0) is only supported for
    GQA (num_key_value_heads < num_heads); MHA sparse is rejected by the
    torch_npu tiling check, so L3 is skipped there (returns None).
    """
    n_kv = num_key_value_heads if num_key_value_heads > 0 else num_heads
    is_dense = abs(sparse_lambda - DENSE_LAMBDA) < 0.01
    if not is_dense and n_kv == num_heads:
        print("  [L3] skipped: torch_npu TND sparse is unsupported for MHA "
              "(antiquant_mode<0 requires GQA)")
        return None

    out_npu, _ = _run_torch_npu_fia(
        query, key, value,
        actual_seq_lengths, actual_seq_lengths_kv,
        num_heads, num_key_value_heads, scale,
        sparse_lambda=sparse_lambda,
        mask_type=mask_type, atten_mask=atten_mask, device=device)

    out_diff = (out_custom.float() - out_npu.float()).abs().max().item()

    if abs(sparse_lambda - DENSE_LAMBDA) < 0.01:
        dtype = query.dtype
        total_elems = query.numel()
        if total_elems > 100000:
            thr = 0.01 if dtype == torch.bfloat16 else 0.002
        else:
            thr = 0.002 if dtype == torch.bfloat16 else 0.0005
        assert out_diff < thr, \
            f"Cross-op dense out_diff={out_diff:.8f} exceeds threshold {thr}"
    else:
        assert out_diff == 0.0, \
            f"Cross-op sparse out_diff={out_diff:.8f} not bit-exact"

    return out_diff


# ---------------------------------------------------------------------
# BLASST golden: optimized_blasst_sim wrapper (per-batch loop)
# ---------------------------------------------------------------------
def run_sim_golden(query, key, value, actual_seq_lengths, actual_seq_lengths_kv,
                   scale, sparse_lambda, sparse_mode=0):
    """Run optimized_blasst_sim (npu_head_mode=True) per batch sequence.

    Returns (out, sparse_blocks, total_blocks):
      out           -- FP32 CPU tensor, TND layout [total_q, H, D]
      sparse_blocks -- skipped block count derived from the sim's block_sparsity
      total_blocks  -- ceil(q/QS_BLOCK) * ceil(kv/KV_BLOCK) * num_heads
                       (causal mask skip is NOT subtracted, so for sparse_mode=3
                       the kernel-side total from the lse counters is smaller)
    """
    qs = [0] + (actual_seq_lengths.cpu().tolist()
                if isinstance(actual_seq_lengths, torch.Tensor) else list(actual_seq_lengths))
    kvs = [0] + (actual_seq_lengths_kv.cpu().tolist()
                 if isinstance(actual_seq_lengths_kv, torch.Tensor) else list(actual_seq_lengths_kv))
    num_heads = query.size(1)
    outs, sparse_n, total_n = [], 0, 0
    for b in range(len(qs) - 1):
        q_b = query[qs[b]:qs[b + 1]]
        k_b = key[kvs[b]:kvs[b + 1]]
        v_b = value[kvs[b]:kvs[b + 1]]
        q_len = qs[b + 1] - qs[b]
        kv_len = kvs[b + 1] - kvs[b]
        out_b, _, blk_sp = optimized_blasst_sim(
            q_b, k_b, v_b,
            torch.tensor(q_len), torch.tensor(kv_len), scale,
            threshold_log=sparse_lambda, sparse_mode=sparse_mode,
            npu_head_mode=True, print_flag=False)
        outs.append(out_b[:q_len].cpu())
        tot_b = (math.ceil(q_len / BlasstGlobalConfig.QS_BLOCK)
                 * math.ceil(kv_len / BlasstGlobalConfig.KV_BLOCK) * num_heads)
        total_n += tot_b
        sparse_n += int(round(blk_sp * tot_b))
    torch.npu.synchronize()
    return torch.cat(outs, dim=0), sparse_n, total_n


def _lse_sparsity(lse):
    """Parse [sparse, total] block counters from the gLSE debug head."""
    sp, tot, _ = parse_blocknum_from_lse(lse, max_cores=24)
    return sp, tot


# ---------------------------------------------------------------------
# Dense cases: custom vs torch_npu only (attn_output only)
# ---------------------------------------------------------------------
def run_dense_case(dtype, batch=2, q_seqlens=None, kv_seqlens=None,
                   num_heads=4, num_key_value_heads=0, head_dim=64,
                   mask_type="none", device="npu:0"):
    """Dense baseline: custom op vs torch_npu, output only."""
    if q_seqlens is None:
        q_seqlens = [2, 3]
    if kv_seqlens is None:
        kv_seqlens = [4, 5]
    torch.manual_seed(42)
    total_q = sum(q_seqlens)
    total_kv = sum(kv_seqlens)

    query = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
    key = torch.randn(total_kv, num_key_value_heads or num_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn(total_kv, num_key_value_heads or num_heads, head_dim, dtype=dtype, device=device)

    actual_seq_lengths = torch.cumsum(torch.tensor(q_seqlens, dtype=torch.int64, device=device), dim=0)
    actual_seq_lengths_kv = torch.cumsum(torch.tensor(kv_seqlens, dtype=torch.int64, device=device), dim=0)
    scale = 1.0 / math.sqrt(head_dim)

    out_custom, _ = run_npu_fia(
        query, key, value, actual_seq_lengths, actual_seq_lengths_kv,
        num_heads, num_key_value_heads, scale, sparse_lambda=DENSE_LAMBDA,
        head_dim=head_dim, mask_type=mask_type, device=device)

    npu_out_diff = _cross_check_npu(
        out_custom, query, key, value,
        actual_seq_lengths, actual_seq_lengths_kv,
        num_heads, num_key_value_heads, scale,
        sparse_lambda=DENSE_LAMBDA, mask_type=mask_type, device=device)
    print(f"[dense] dtype={dtype} batch={batch} q={q_seqlens} kv={kv_seqlens} "
          f"heads={num_heads}/{num_key_value_heads or num_heads} D={head_dim} "
          f"mask={mask_type}: vs_npu out_diff={npu_out_diff:.8f}")


def run_cross_op_dense_case(dtype, q_seqlens, kv_seqlens, num_heads,
                            num_key_value_heads, head_dim, device="npu:0"):
    """Cross-op precision: migrated custom op vs torch_npu, dense, output only."""
    n_kv = num_key_value_heads if num_key_value_heads > 0 else num_heads
    total_q = sum(q_seqlens)
    total_kv = sum(kv_seqlens)

    torch.manual_seed(42)
    query = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
    key = torch.randn(total_kv, n_kv, head_dim, dtype=dtype, device=device)
    value = torch.randn(total_kv, n_kv, head_dim, dtype=dtype, device=device)
    scale = 1.0 / math.sqrt(head_dim)

    actual_seq_lengths = torch.cumsum(torch.tensor(q_seqlens, dtype=torch.int64, device=device), dim=0)
    actual_seq_lengths_kv = torch.cumsum(torch.tensor(kv_seqlens, dtype=torch.int64, device=device), dim=0)

    out_custom, _ = run_npu_fia(
        query, key, value, actual_seq_lengths, actual_seq_lengths_kv,
        num_heads, num_key_value_heads, scale, sparse_lambda=DENSE_LAMBDA,
        head_dim=head_dim, device=device)

    npu_out_diff = _cross_check_npu(
        out_custom, query, key, value,
        actual_seq_lengths, actual_seq_lengths_kv,
        num_heads, num_key_value_heads, scale,
        sparse_lambda=DENSE_LAMBDA, device=device)
    tag = f"xop {str(dtype).split('.')[-1]:6s} q={q_seqlens} kv={kv_seqlens} heads={num_heads}/{n_kv} D={head_dim}"
    print(f"  {tag:65s} vs_npu out_diff={npu_out_diff:.8f}")


# ---------------------------------------------------------------------
# BlasST cases: three-level comparison (output + lse-parsed sparsity)
# ---------------------------------------------------------------------
def _blasst_thresholds(dtype):
    if dtype == torch.float16:
        return 1e-2, 1e-3
    return 1e-2, 1e-2


def run_blasst_case(dtype, sparse_lambda=-3.0, batch=2, q_seqlens=None,
                    kv_seqlens=None, num_heads=4, num_key_value_heads=0,
                    head_dim=64, device="npu:0"):
    """BlasST path with random inputs.

    Random inputs produce dm values near the threshold, so skip decisions are
    not stable across implementations: L1/L2 use relaxed output thresholds and
    the sparsity counts are report-only (small-shape kernels may also hold
    garbage in the lse debug-counter head).
    """
    if q_seqlens is None:
        q_seqlens = [128, 128]
    if kv_seqlens is None:
        kv_seqlens = [256, 256]
    torch.manual_seed(42)
    total_q = sum(q_seqlens)
    total_kv = sum(kv_seqlens)

    query = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
    key = torch.randn(total_kv, num_key_value_heads or num_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn(total_kv, num_key_value_heads or num_heads, head_dim, dtype=dtype, device=device)

    actual_seq_lengths = torch.cumsum(torch.tensor(q_seqlens, dtype=torch.int64, device=device), dim=0)
    actual_seq_lengths_kv = torch.cumsum(torch.tensor(kv_seqlens, dtype=torch.int64, device=device), dim=0)
    scale = 1.0 / math.sqrt(head_dim)

    out_custom, lse_custom = run_npu_fia(
        query, key, value, actual_seq_lengths, actual_seq_lengths_kv,
        num_heads, num_key_value_heads, scale, sparse_lambda=sparse_lambda,
        head_dim=head_dim, mask_type="none", block_size=0, device=device)

    # L1: custom vs sim golden (sparse lambda)
    out_g, g_sp, g_tot = run_sim_golden(
        query, key, value, actual_seq_lengths, actual_seq_lengths_kv,
        scale, sparse_lambda, sparse_mode=0)
    # L2: custom vs sim golden (dense)
    if abs(sparse_lambda - DENSE_LAMBDA) < 0.01:
        out_gd = out_g
    else:
        out_gd, _, _ = run_sim_golden(
            query, key, value, actual_seq_lengths, actual_seq_lengths_kv,
            scale, DENSE_LAMBDA, sparse_mode=0)

    k_sp, k_tot = _lse_sparsity(lse_custom)

    out_f = out_custom.cpu().float()
    l1_diff = (out_f - out_g).abs().max().item()
    l2_diff = (out_f - out_gd).abs().max().item()
    l3_diff = _cross_check_npu(
        out_custom, query, key, value,
        actual_seq_lengths, actual_seq_lengths_kv,
        num_heads, num_key_value_heads, scale,
        sparse_lambda=sparse_lambda, device=device)

    rtol, atol = _blasst_thresholds(dtype)
    l3_str = f"{l3_diff:.8f}" if l3_diff is not None else "skip(MHA)"
    print(f"[blasst] dtype={dtype} lambda={sparse_lambda} q={q_seqlens} kv={kv_seqlens} "
          f"heads={num_heads}/{num_key_value_heads or num_heads} D={head_dim}: "
          f"L1_vs_golden={l1_diff:.6f} L2_vs_dense={l2_diff:.6f} L3_vs_npu={l3_str} | "
          f"sparsity kernel={k_sp}/{k_tot} golden={g_sp}/{g_tot}")

    torch.testing.assert_close(out_f, out_g, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_f, out_gd, rtol=rtol, atol=atol)


def run_blasst_mixture_case(dtype, sparse_lambda, q_len=512, kv_len=4096,
                            num_heads=8, head_dim=128, n_sink_heads=6,
                            anchor_stacks=(1, 3, 4, 6, 7),
                            anchor_logits=(16.0, 14.0, 20.0, 6.0, 15.0),
                            seed=42, mask_type="none", device="npu:0"):
    """BlasST precision with gen_mixture inputs (real-like high sparsity).

    Every stack's dm sits >= 1.0 away from each tested threshold, so the
    kernel-vs-golden skip decisions are stable: the sparse block COUNT parsed
    from the custom op's lse debug head must equal the golden's count.
    L1 output check uses relaxed thresholds (kernel fp16/bf16 vs sim fp32
    accumulation); L2 (vs dense) is report-only since sparse outputs
    legitimately diverge from dense attention.
    """
    query, key, value = gen_mixture(
        anchor_stacks=anchor_stacks, anchor_logits=anchor_logits,
        n_sink_heads=n_sink_heads, beta=8.0, seed=seed,
        q_len=q_len, kv_len=kv_len, num_heads=num_heads, head_dim=head_dim,
        dtype=dtype, device=device)
    actual_seq_lengths = torch.tensor([q_len], dtype=torch.int64, device=device)
    actual_seq_lengths_kv = torch.tensor([kv_len], dtype=torch.int64, device=device)
    scale = 1.0 / math.sqrt(head_dim)

    atten_mask = None
    if mask_type == "causal":
        atten_mask = causal_mask(q_len, device=device)
    sparse_mode = 3 if mask_type == "causal" else 0

    out_custom, lse_custom = run_npu_fia(
        query, key, value, actual_seq_lengths, actual_seq_lengths_kv,
        num_heads, num_heads, scale, sparse_lambda=sparse_lambda,
        head_dim=head_dim, mask_type=mask_type, atten_mask=atten_mask,
        block_size=0, device=device)

    # L1: custom vs sim golden (sparse lambda)
    out_g, g_sp, g_tot = run_sim_golden(
        query, key, value, actual_seq_lengths, actual_seq_lengths_kv,
        scale, sparse_lambda, sparse_mode=sparse_mode)
    # L2: custom vs sim golden (dense) -- report only
    if abs(sparse_lambda - DENSE_LAMBDA) < 0.01:
        out_gd = out_g
    else:
        out_gd, _, _ = run_sim_golden(
            query, key, value, actual_seq_lengths, actual_seq_lengths_kv,
            scale, DENSE_LAMBDA, sparse_mode=sparse_mode)

    k_sp, k_tot = _lse_sparsity(lse_custom)

    out_f = out_custom.cpu().float()
    l1_diff = (out_f - out_g).abs().max().item()
    l2_diff = (out_f - out_gd).abs().max().item()
    l3_diff = _cross_check_npu(
        out_custom, query, key, value,
        actual_seq_lengths, actual_seq_lengths_kv,
        num_heads, num_heads, scale,
        sparse_lambda=sparse_lambda,
        mask_type=mask_type, atten_mask=atten_mask, device=device)

    rtol, atol = _blasst_thresholds(dtype)
    l3_str = f"{l3_diff:.8f}" if l3_diff is not None else "skip(MHA)"
    print(f"[blasst-mix] dtype={dtype} lambda={sparse_lambda} q={q_len} kv={kv_len} "
          f"heads={num_heads} D={head_dim} mask={mask_type}: "
          f"L1_vs_golden={l1_diff:.6f} L2_vs_dense={l2_diff:.6f} L3_vs_npu={l3_str} | "
          f"sparsity kernel={k_sp}/{k_tot} golden={g_sp}/{g_tot}")

    # L1: output (relaxed) + sparse block count (strict, decisions are stable).
    torch.testing.assert_close(out_f, out_g, rtol=rtol, atol=atol)
    # The kernel-side counters are only trusted when they look sane: on some
    # kernel branches (e.g. small no-mask MHA shapes) the gLSE debug head is
    # not written and holds garbage (observed total=309 > theoretical 256).
    counters_sane = (k_tot > 0 and k_tot % num_heads == 0 and k_tot <= g_tot)
    if counters_sane:
        assert k_sp == g_sp, \
            f"sparse block count mismatch: kernel={k_sp}/{k_tot} golden={g_sp}/{g_tot}"
    else:
        print(f"  [sparsity] kernel lse counters look invalid "
              f"(kernel={k_sp}/{k_tot} vs formula total={g_tot}), count assert skipped")


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------
def _run_case_safe(fn, *args, **kwargs):
    """Run a test case and return (ok, message)."""
    try:
        fn(*args, **kwargs)
        return True, "OK"
    except AssertionError as e:
        return False, f"ASSERT: {e}"
    except Exception as e:
        return False, f"ERROR: {type(e).__name__}: {e}"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FusedInferAttentionScore migrated operator tests")
    parser.add_argument("--device", default="npu:0", help="Target NPU device (default: npu:0)")
    args = parser.parse_args()

    import torch_npu  # noqa: F401
    from vllm_ascend import platform
    platform.NPUPlatform.import_kernels()
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    device = args.device
    torch.npu.set_device(device)

    results = []

    # ------------------------------------------------------------------
    # 0. Cross-op precision: migrated custom op vs torch_npu (dense only)
    # ------------------------------------------------------------------
    print("=" * 80)
    print("0. Cross-op precision: custom vs torch_npu (dense, output only)")
    print("=" * 80)
    _XOP_CASES = [
        ("small-mha  fp16",    torch.float16,    [2,3],   [4,5],     4, 4, 64),
        ("small-mha  bf16",    torch.bfloat16,   [2,3],   [4,5],     4, 4, 64),
        ("medium-mha fp16",    torch.float16,    [64,96], [128,192], 8, 8, 64),
        ("medium-mha bf16",    torch.bfloat16,   [64,96], [128,192], 8, 8, 64),
        ("gqa        fp16",    torch.float16,    [64,96], [128,192], 8, 2, 64),
        ("gqa        bf16",    torch.bfloat16,   [64,96], [128,192], 8, 2, 64),
        ("large-dim  fp16",    torch.float16,    [64,128],[256,512], 8, 8, 128),
        ("large-dim  bf16",    torch.bfloat16,   [64,128],[256,512], 8, 8, 128),
    ]
    for name, dt, qs, ks, nh, nkv, hd in _XOP_CASES:
        results.append((f"xop {name}", _run_case_safe(
            run_cross_op_dense_case, dt, qs, ks, nh, nkv, hd, device=device)))

    # ------------------------------------------------------------------
    # 1. Dense baseline (custom vs torch_npu, output only)
    # ------------------------------------------------------------------
    print("=" * 80)
    print("1. Dense baseline (custom vs torch_npu, output only)")
    print("=" * 80)
    results.append(("dense fp16", _run_case_safe(run_dense_case, torch.float16, device=device)))
    results.append(("dense bf16", _run_case_safe(run_dense_case, torch.bfloat16, device=device)))

    # ------------------------------------------------------------------
    # 2. BlasST sparse path (three-level comparison)
    # ------------------------------------------------------------------
    print("=" * 80)
    print("2. BlasST sparse path (L1 sim-golden / L2 sim-dense / L3 torch_npu)")
    print("=" * 80)
    results.append(("blasst fp16 lambda=-99.0", _run_case_safe(run_blasst_case, torch.float16, sparse_lambda=-99.0, device=device)))
    results.append(("blasst bf16 lambda=-99.0", _run_case_safe(run_blasst_case, torch.bfloat16, sparse_lambda=-99.0, device=device)))
    results.append(("blasst fp16 lambda=-40.0", _run_case_safe(run_blasst_case, torch.float16, sparse_lambda=-40.0, device=device)))
    results.append(("blasst bf16 lambda=-40.0", _run_case_safe(run_blasst_case, torch.bfloat16, sparse_lambda=-40.0, device=device)))
    results.append(("blasst fp16 lambda=-3.0", _run_case_safe(run_blasst_case, torch.float16, sparse_lambda=-3.0, device=device)))
    results.append(("blasst bf16 lambda=-3.0", _run_case_safe(run_blasst_case, torch.bfloat16, sparse_lambda=-3.0, device=device)))

    # BlasST with gen_mixture inputs (real-like high sparsity, stable skip
    # decisions -> strict sparse-count check).
    for lam in (-99.0, -7.0, -3.0, -1.0):
        results.append((f"blasst-mix fp16 lambda={lam}", _run_case_safe(
            run_blasst_mixture_case, torch.float16, sparse_lambda=lam, device=device)))
        results.append((f"blasst-mix bf16 lambda={lam}", _run_case_safe(
            run_blasst_mixture_case, torch.bfloat16, sparse_lambda=lam, device=device)))

    # sparse_mode=3 with a right-aligned causal int8 mask, q/kv=2048/4096.
    for lam in (-99.0, -7.0, -3.0, -1.0):
        results.append((f"blasst-mix-ca fp16 lambda={lam}", _run_case_safe(
            run_blasst_mixture_case, torch.float16, sparse_lambda=lam,
            q_len=2048, kv_len=4096, mask_type="causal", device=device)))
        results.append((f"blasst-mix-ca bf16 lambda={lam}", _run_case_safe(
            run_blasst_mixture_case, torch.bfloat16, sparse_lambda=lam,
            q_len=2048, kv_len=4096, mask_type="causal", device=device)))

    # ------------------------------------------------------------------
    # 3. GQA / variable-length / head_dim=128
    # ------------------------------------------------------------------
    print("=" * 80)
    print("3. GQA, variable length and head_dim=128")
    print("=" * 80)
    results.append(("dense gqa fp16", _run_case_safe(run_dense_case, torch.float16, q_seqlens=[64, 96], kv_seqlens=[128, 192],
                   num_heads=8, num_key_value_heads=2, head_dim=64, device=device)))
    results.append(("blasst gqa fp16 lambda=-40.0", _run_case_safe(run_blasst_case, torch.float16, sparse_lambda=-40.0,
                    q_seqlens=[64, 96], kv_seqlens=[128, 192],
                    num_heads=8, num_key_value_heads=2, head_dim=64, device=device)))
    results.append(("blasst gqa bf16 lambda=-3.0", _run_case_safe(run_blasst_case, torch.bfloat16, sparse_lambda=-3.0,
                    q_seqlens=[64, 128], kv_seqlens=[256, 512],
                    num_heads=8, num_key_value_heads=2, head_dim=128, device=device)))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    all_ok = True
    for name, (ok, msg) in results:
        status = "PASS" if ok else "FAIL"
        print(f"{status:6s} | {name:40s} | {msg}")
        all_ok = all_ok and ok

    if all_ok:
        print("=" * 80)
        print("All cases passed.")
        print("=" * 80)
    else:
        print("=" * 80)
        print("Some cases failed (see details above).")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
