#!/usr/bin/env python3
"""跳算版 custom kernel vs 稀疏跳过 golden（kernel 粒度）单算子对比。

golden = BlasstGoldenTND(golden_mode="kernel")：qSBlock=128 × head × KV stack 512，
两个 64 行 subBlock × 16 行 rowLoop —— 与 kernel 跳算粒度一致、且实现稀疏跳过语义。

对比矩阵：lambda ∈ {-7, -5, -3, -1}：
  - custom(-λ, 真跳过)  vs golden(λ, 稀疏跳过)     —— 跳算语义一致性
  - custom(-99, 不跳)   vs golden(-99, dense)      —— dense 基线 sanity
  - custom stats 检测率 vs golden block_sparsity   —— 检测口径交叉验证
"""

import os
import sys

import torch

_CUR = os.path.dirname(os.path.realpath(__file__))
_CUSTOM_OPP = os.path.join(_CUR, "..", "..", "vllm_ascend", "_cann_ops_custom", "vendors", "vllm-ascend")
if os.path.exists(_CUSTOM_OPP):
    os.environ.setdefault("ASCEND_CUSTOM_OPP_PATH", _CUSTOM_OPP)

import torch_npu  # noqa: F401
import vllm_ascend  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

sys.path.insert(0, _CUR)
from run_dump_repro import _to_tensor, load_dump, run_custom_op  # noqa: E402
from blasst_golden_tnd import BlasstGoldenTND  # noqa: E402


def build_golden_inputs(inputs, NH, NKV, SC, BS):
    """paged dump → golden 所需 TND 张量（正确处理多 kv head）。"""
    query = inputs["query"].cpu()
    D = query.size(-1)
    q_cum = inputs["actual_seq_lengths_q"].cpu().long()
    kv_lens = inputs["actual_seq_lengths_kv"].cpu().tolist()  # paged: per-batch
    bt = inputs["block_table"].cpu()

    def depag(cache):
        chunks = []
        for b, kv_len in enumerate(kv_lens):
            nb = (kv_len + BS - 1) // BS
            idx = bt[b, :nb].long()
            t = cache[idx]                          # [nb, bs, kvh*D]
            t = t.reshape(-1, NKV, D)[:kv_len]      # [kv_len, kvh, D]
            chunks.append(t)
        return torch.cat(chunks, 0)

    key_tnd = depag(inputs["key"].cpu())
    value_tnd = depag(inputs["value"].cpu())
    kv_cum = torch.cumsum(torch.tensor(kv_lens, dtype=torch.int64), dim=0)

    am = inputs["atten_mask"]
    masks, q_start = [], 0
    for b in range(q_cum.numel()):
        q_len = int(q_cum[b].item()) - q_start
        q_start = int(q_cum[b].item())
        kv_len = kv_lens[b]
        if isinstance(am, torch.Tensor) and q_len == am.size(0) and kv_len >= am.size(1):
            # kernel 语义：mask 覆盖最后 am.size(1) 列（右对齐窗口），
            # 之前的 KV 列可见（无 mask）→ 左侧补 0
            left = kv_len - am.size(1)
            m = torch.zeros(q_len, kv_len, dtype=torch.int8)
            if left > 0:
                m[:, left:] = am.cpu()[:q_len]
            else:
                m = am.cpu()[:q_len].to(torch.int8).clone()
            masks.append(m)
        else:
            masks.append(None)
    return query, key_tnd, value_tnd, q_cum, kv_cum, masks

CASES = [
    ("prefill_l30", "/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps_v2/fia_layer30_rank0_chunked_prefill_occ1.pt"),
    ("decode_l20", "/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps_v2/fia_layer20_rank0_decode_bigdiff_occ0.pt"),
]
LAMBDAS = [-7.0, -5.0, -3.0, -1.0]


def cmp(a, b):
    a = a.cpu().float()
    b = b.float()
    if a.shape != b.shape:
        # kernel 输出可能带 padding 行（Q 对齐），按有效行比较
        rows = min(a.shape[0], b.shape[0])
        a, b = a[:rows], b[:rows]
    d = (a - b).abs()
    return d.max().item(), d.mean().item()


def main():
    print("=" * 100)
    print("跳算版 custom vs 稀疏跳过 golden（kernel 粒度）")
    print("=" * 100)
    for case_name, path in CASES:
        inputs, params, _, _, meta = load_dump(path)
        NH, NKV = params["num_heads"], params["num_kv_heads"]
        SC, BS = params["scale"], params["block_size"]
        sm = int(params.get("sparse_mode", 3))
        dev = "npu:0"

        q = inputs["query"].to(dev)
        k = inputs["key"].to(dev)
        v = inputs["value"].to(dev)
        am = inputs["atten_mask"]
        am = am.to(dev) if isinstance(am, torch.Tensor) and am.numel() > 0 else None
        aq = _to_tensor(inputs["actual_seq_lengths_q"], dev)
        akv = _to_tensor(inputs["actual_seq_lengths_kv"], dev)
        bt = inputs["block_table"]
        bt = bt.to(dev) if isinstance(bt, torch.Tensor) else None
        if sm == 3 and am is None:
            sm = 0

        # warmup 规避首调用 gSp 冷启动竞态（已知间歇性 NaN bug，另行排查）
        for _ in range(3):
            run_custom_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, -99.0,
                          lse_flag=False, stats_flag=False)
        torch.npu.synchronize()

        print(f"\ncase: {case_name}  layer={meta['layer']}  q={list(q.shape)}")
        print(f"{'lambda':>7} | {'cust_skip vs gold_skip':>22} | {'cust_dense vs gold_dense':>24} | "
              f"{'cust_stats稀疏率':>14} | {'golden稀疏率':>11}")

        # golden 输入：paged → 连续 TND（多 kv head 正确去分页）
        gq, gk, gv, q_cum, kv_cum, masks = build_golden_inputs(inputs, NH, NKV, SC, BS)
        golden = BlasstGoldenTND(
            num_heads=NH, num_key_value_heads=NKV, head_dim=gq.size(-1),
            scale=SC, block_size=BS, sparse_lamda=-99.0)

        def run_gold(lam):
            golden.sparse_lamda = lam
            out_ref, _, info = golden.forward_blasst_kernel(gq, gk, gv, q_cum, kv_cum, masks=masks)
            return out_ref, info

        # dense 基线
        out_d, _, _ = run_custom_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, -99.0,
                                    lse_flag=False, stats_flag=False)
        torch.npu.synchronize()
        out_gd, info_d = run_gold(-99.0)
        dmx, dmn = cmp(out_d, out_gd)
        print(f"  {-99.0:>7.1f} | {'n/a':>22} | {f'{dmx:.4e} / {dmn:.2e}':>24} | {'0.00%':>14} | "
              f"{str(info_d):>11}")

        for lam in LAMBDAS:
            out_s, _, _ = run_custom_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, lam,
                                        lse_flag=False, stats_flag=False)
            torch.npu.synchronize()
            out_gs, info_s = run_gold(lam)
            smx, smn = cmp(out_s, out_gs)
            # kernel 检测率（stats 模式）
            _, _, ss = run_custom_op(q, k, v, am, aq, akv, bt, NH, NKV, SC, sm, BS, lam,
                                     lse_flag=False, stats_flag=True)
            torch.npu.synchronize()
            vv = ss.cpu().flatten()
            sp, tot = int(vv[0]), int(vv[1])
            crate = 100.0 * sp / tot if tot > 0 else 0.0
            print(f"  {lam:>7.1f} | {f'{smx:.4e} / {smn:.2e}':>22} | {'n/a':>24} | "
                  f"{f'{crate:.2f}% ({sp}/{tot})':>14} | {str(info_s):>11}")

    print("\n" + "=" * 100)
    print("完成")
    print("=" * 100)


if __name__ == "__main__":
    main()
