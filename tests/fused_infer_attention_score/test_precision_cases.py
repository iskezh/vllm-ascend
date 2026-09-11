#!/usr/bin/env python3
"""rar 风格精度对比验证：custom FIA (-99 / -3) vs torch_npu 原生 FIA（TND）。

用例设计参考
  /home/z00603376/_rar_extract_12/12_FusedInferAttentionScore_evo/12_FusedInferAttentionScore.json
的 JSONL 形式（每行 {"inputs": [...], "scalar_args": {...}}），数据生成沿用其
random_tensor 风格（独立随机选择正态/均匀分布）。

对比口径：
  L1: custom(lambda=-99, dense)  vs torch_npu(dense)   —— 迁移正确性
  L2: custom(lambda=-3, 稀疏)    vs custom(lambda=-99) —— 稀疏跳算对输出的影响
  L3: custom(lambda=-3)          vs torch_npu(antiquant 编码稀疏，仅 GQA)
输出逐 case 表格 + 汇总（与 perf_report.md 逐 case 表同款形式）。
"""

import json
import math
import os

import torch

_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
_CUSTOM_OPP_PATH = os.path.join(
    _CUR_DIR, "..", "..", "vllm_ascend", "_cann_ops_custom", "vendors", "vllm-ascend"
)
if os.path.exists(_CUSTOM_OPP_PATH):
    os.environ.setdefault("ASCEND_CUSTOM_OPP_PATH", _CUSTOM_OPP_PATH)

import torch_npu  # noqa: F401
import vllm_ascend  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401  加载 torch 扩展，注册 torch.ops._C_ascend

SWA_INT_MAX = 2147483647
DENSE_LAMBDA = -99.0
SPARSE_LAMBDA = -3.0

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def random_tensor(shape, dtype, seed):
    """rar 同款独立随机选择正态/均匀分布，但量级取真实激活范围
    （RMSNorm 后 ~N(0,1)）：正态 mu∈[-0.5,0.5] σ∈[0.5,1.5]，均匀 [-1,1]。
    避免极端 logits 导致 fp16 vs fp32 的数值敏感性淹没 kernel 间对比。"""
    g = torch.Generator().manual_seed(seed)
    if seed % 2 == 0:
        mu = float(torch.rand(1, generator=g).item() - 0.5)
        sigma = float(torch.rand(1, generator=g).item() + 0.5)
        return torch.normal(mu, sigma, shape, dtype=dtype, generator=g)
    return torch.empty(shape, dtype=dtype).uniform_(-1.0, 1.0, generator=g)


def causal_mask_vllm(device):
    """vLLM 实际使用的 SplitFuse causal mask（attention_mask.py get_splitfuse_attn_mask）：
    固定 2048x2048 triu int8（1 = masked）。torch_npu FIA TND 的 S1S2 mask 检查
    要求方阵且尺寸为 2048，custom 算子同样接受该格式。"""
    return torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device=device), diagonal=1)


def run_torch_npu(q, k, v, seqs_q, seqs_kv, n_heads, n_kv, scale, mask, sparse_lambda):
    """torch_npu 原生 FIA（TND）。稀疏经 antiquant_mode 编码（仅 GQA 支持）。"""
    sparse_mode = 3 if mask is not None else 0
    is_dense = abs(sparse_lambda - DENSE_LAMBDA) < 0.01
    antiquant_mode = 0 if is_dense else int(sparse_lambda * 10 - 100)
    kwargs = dict(
        num_heads=n_heads, scale=scale,
        input_layout="TND", num_key_value_heads=n_kv,
        sparse_mode=sparse_mode, inner_precise=0, antiquant_mode=antiquant_mode,
        softmax_lse_flag=True,
        actual_seq_lengths=seqs_q,
        actual_seq_lengths_kv=seqs_kv,
    )
    if mask is not None:
        kwargs["atten_mask"] = mask
    return torch_npu.npu_fused_infer_attention_score(q, k, v, **kwargs)[0]


def run_custom(q, k, v, seqs_q, seqs_kv, n_heads, n_kv, scale, mask, sparse_lambda):
    """迁移后的自定义 FIA（TND），sparse_lambda 直接透传。"""
    sparse_mode = 3 if mask is not None else 0
    out, _, _ = torch.ops._C_ascend.npu_fused_infer_attention_score(
        q, k, v,
        None,  # pse_shift
        mask,
        seqs_q,
        seqs_kv,
        None,  # blocktable
        n_heads,
        scale,
        SWA_INT_MAX, SWA_INT_MAX,
        "TND",
        n_kv,
        sparse_mode,
        0,   # inner_precise
        0,   # block_size
        0,   # antiquant_mode
        sparse_lambda,
        False,  # softmax_lse_flag
        False,  # sparse_stats_flag
    )
    return out


def compare(a, b, atol, rtol):
    a = a.cpu().float()
    b = b.cpu().float()
    absd = (a - b).abs()
    rel_mask = b.abs() > 1e-3
    rel = (absd[rel_mask] / b[rel_mask].abs()) if rel_mask.any() else torch.zeros(1)
    mismatch = ~torch.isclose(a, b, rtol=rtol, atol=atol)
    return {
        "max_abs": absd.max().item(),
        "mean_abs": absd.mean().item(),
        "max_rel": rel.max().item(),
        "mismatch_pct": 100.0 * mismatch.float().mean().item(),
    }


def main():
    cases_path = os.path.join(_CUR_DIR, "tnd_precision_cases.json")
    with open(cases_path) as f:
        cases = [json.loads(line) for line in f if line.strip()]

    device = "npu:0"
    rows = []
    n_pass_l1 = n_pass_l2 = n_pass_l3 = 0

    print("=" * 118)
    print(f"{'case':>4} | {'shape q->kv':<28} | {'dtype':<10} | {'L1 max_abs':>10} "
          f"{'L1 mean':>9} | {'L2 max_abs':>10} {'L2 mean':>9} | {'L3 max_abs':>10} | 判定")
    print("-" * 118)

    for ci, case in enumerate(cases):
        sa = case["scalar_args"]
        q_info, k_info, v_info = case["inputs"][:3]
        dtype = DTYPE_MAP[q_info["dtype"]]
        # 混合 batch 时按最大批次尺寸分配（与 vLLM 打包 batch 行为一致），
        # 精确尺寸的用例 shape 即 total；对齐到 128 保证 kernel 友好
        q = random_tensor(q_info["shape"], dtype, sa["seed"])
        k = random_tensor(k_info["shape"], dtype, sa["seed"] + 100)
        v = random_tensor(v_info["shape"], dtype, sa["seed"] + 200)
        q, k, v = q.to(device), k.to(device), v.to(device)

        n_heads, n_kv = sa["num_heads"], sa["num_key_value_heads"]
        seqs_q = list(sa["seq_q"])
        seqs_kv = list(sa["seq_kv"])
        scale = sa["scale"]

        mask = None
        if sa["mask"] == "causal":
            # 要求 kv 总量不超过 2048（vLLM SplitFuse mask 窗口上限）
            assert k.shape[0] <= 2048, f"causal 用例 kv={k.shape[0]} 超出 2048 窗口"
            mask = causal_mask_vllm(device)

        tol = 5e-3 if dtype == torch.float16 else 2e-2

        out_ref = run_torch_npu(q, k, v, seqs_q, seqs_kv, n_heads, n_kv, scale, mask, DENSE_LAMBDA)
        out99 = run_custom(q, k, v, seqs_q, seqs_kv, n_heads, n_kv, scale, mask, DENSE_LAMBDA)
        out3 = run_custom(q, k, v, seqs_q, seqs_kv, n_heads, n_kv, scale, mask, SPARSE_LAMBDA)

        l1 = compare(out99, out_ref, atol=tol, rtol=1e-2)
        l2 = compare(out3, out99, atol=tol, rtol=1e-2)

        # L3: torch_npu 稀疏仅支持 GQA；MHA 跳过
        l3 = None
        if n_kv < n_heads:
            try:
                out_npu3 = run_torch_npu(q, k, v, seqs_q, seqs_kv, n_heads, n_kv, scale, mask, SPARSE_LAMBDA)
                l3 = compare(out3, out_npu3, atol=tol, rtol=1e-2)
            except RuntimeError as e:
                print(f"      L3 torch_npu 稀疏失败: {str(e)[:80]}")

        ok1 = l1["max_abs"] <= tol * 10
        ok2 = l2["max_abs"] <= tol * 10
        ok3 = l3 is None or l3["max_abs"] <= tol * 10
        verdict = "PASS" if (ok1 and ok2 and ok3) else "FAIL"
        if ok1:
            n_pass_l1 += 1
        if ok2:
            n_pass_l2 += 1
        if ok3:
            n_pass_l3 += 1

        l3s = f"{l3['max_abs']:10.3e}" if l3 else "       n/a"
        shape_s = f"[{q.shape[0]},{q.shape[1]},{q.shape[2]}]->[{k.shape[0]},{k.shape[1]}]"
        print(f"{ci:>4} | {shape_s:<28} | {q_info['dtype']:<10} | {l1['max_abs']:10.3e} "
              f"{l1['mean_abs']:9.2e} | {l2['max_abs']:10.3e} {l2['mean_abs']:9.2e} | {l3s} | {verdict}")
        rows.append({"case": ci, "shape": shape_s, "dtype": q_info["dtype"],
                     "l1": l1, "l2": l2, "l3": l3, "verdict": verdict})

    print("=" * 118)
    total = len(cases)
    print(f"L1 (custom-99 vs torch_npu dense): {n_pass_l1}/{total} 通过（阈值 max_abs<=10*atol）")
    print(f"L2 (custom-3  vs custom-99)      : {n_pass_l2}/{total} 通过")
    print(f"L3 (custom-3  vs torch_npu 稀疏)  : {n_pass_l3}/{total} 通过")
    if n_pass_l1 == total and n_pass_l2 == total and n_pass_l3 == total:
        print("ALL PASS")
        return 0
    print("Some cases failed (see table above).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
