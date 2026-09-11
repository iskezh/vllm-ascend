#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四方对比: 线上 custom/base 输出 vs 离线 custom/base 重放 (同一份 dump 输入)。

回答: custom 离线为何复现不出线上结果 —— 输入是否保真 (online_base vs offline_base),
以及线上 custom 输出发散的空间模式。
结果落盘: fia_online_offline_compare.txt (同目录)。
"""

import os

_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
_CUSTOM_OPP_PATH = os.path.join(
    _CUR_DIR, "..", "..", "vllm_ascend", "_cann_ops_custom", "vendors",
    "vllm-ascend")
if os.path.exists(_CUSTOM_OPP_PATH):
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = _CUSTOM_OPP_PATH

import torch
import torch_npu  # noqa: F401

from vllm_ascend import platform

platform.NPUPlatform.import_kernels()
import vllm_ascend.vllm_ascend_C  # noqa: F401

DUMP_DIR = '/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps_v2'
DUMP = f'{DUMP_DIR}/fia_layer20_rank0_decode_bigdiff_occ0.pt'
SNAP = f'{DUMP_DIR}/fia_bigdiff_outputs_pid22312_model_layers_20_self_attn_attn.pt'
OUT_TXT = os.path.join(_CUR_DIR, 'fia_online_offline_compare.txt')

_lines = []


def out(s):
    print(s, flush=True)
    _lines.append(s)


def ulp_diff(a, b):
    ai = a.view(torch.uint16).to(torch.int32)
    bi = b.view(torch.uint16).to(torch.int32)
    am_ = torch.where(ai >= 0x8000, -ai, ai)
    bm = torch.where(bi >= 0x8000, -bi, bi)
    u = (am_ - bm).abs()
    d = (a.float() - b.float()).abs()
    return {
        'bitdiff': int((u > 0).sum()), 'ulp>=5': int((u >= 5).sum()),
        'max_abs': d.max().item(), 'numel': a.numel(),
        'per_head_max': d.amax(dim=(0, 2)).tolist(),
        'per_head_bitdiff': (u > 0).any(dim=(0, 2)).tolist(),
    }


def main():
    d = torch.load(DUMP, map_location='cpu', weights_only=False)
    snap = torch.load(SNAP, map_location='cpu', weights_only=False)
    inp, params = d['inputs'], d['params']
    online_c, online_b = snap['output_custom'], snap['output_base']
    dev = torch.device('npu:0')
    torch.npu.set_device(dev)
    with torch.npu.device(dev):
        q = inp['query'].to(dev)
        k = inp['key'].to(dev)
        v = inp['value'].to(dev)
        bt = inp['block_table'].to(dev)
        aq = inp['actual_seq_lengths_q'].cpu().tolist()
        akv = inp['actual_seq_lengths_kv'].cpu().tolist()
        am = inp['atten_mask']
        am = am.to(dev) if isinstance(am, torch.Tensor) and am.numel() > 0 else None
        off_c, _, __ = torch.ops._C_ascend.npu_fused_infer_attention_score(
            q, k, v, None, am, aq, akv, bt, params['num_heads'], params['scale'],
            2147483647, 2147483647, 'TND', params['num_kv_heads'],
            params['sparse_mode'], 0, params['block_size'], 0,
            params['sparse_lambda'], False)
        torch.npu.synchronize()
        off_b, _ = torch_npu.npu_fused_infer_attention_score(
            query=q, key=k, value=v, atten_mask=am, block_table=bt,
            input_layout='TND', block_size=params['block_size'],
            actual_seq_lengths=inp['actual_seq_lengths_q'].cpu().tolist(),
            actual_seq_lengths_kv=inp['actual_seq_lengths_kv'].cpu().tolist(),
            num_key_value_heads=params['num_kv_heads'],
            num_heads=params['num_heads'], scale=params['scale'],
            sparse_mode=params['sparse_mode'])
        torch.npu.synchronize()
    off_c, off_b = off_c.cpu(), off_b.cpu()

    out(f'dump={os.path.basename(DUMP)} snap={os.path.basename(SNAP)}')
    out(f'q_stride(meta)={d.get("q_stride")} q_contiguous={d.get("q_contiguous")} '
        f'bt_meta={d.get("bt_meta")}')
    pairs = [
        ('online_custom vs offline_custom', online_c, off_c),
        ('online_base   vs offline_base  ', online_b, off_b),
        ('online_custom vs online_base   ', online_c, online_b),
        ('offline_custom vs offline_base ', off_c, off_b),
        ('online_custom vs offline_base  ', online_c, off_b),
    ]
    for name, a, b in pairs:
        r = ulp_diff(a, b)
        out(f'[{name}] bitdiff={r["bitdiff"]}/{r["numel"]} ulp>=5:{r["ulp>=5"]} '
            f'max_abs={r["max_abs"]:.4e}')
        out(f'    per-head max: ' +
            ' '.join(f'h{i}={v:.3e}' for i, v in enumerate(r['per_head_max'])))
        out(f'    per-head any-bitdiff: {r["per_head_bitdiff"]}')

    # 线上 custom 发散的空间分布: 逐 token
    d_tok = (online_c.float() - off_c.float()).abs().amax(dim=(1, 2))
    out('online_custom 逐 token maxdiff: ' +
        ' '.join(f't{i}={v:.3e}' for i, v in enumerate(d_tok.tolist())))

    with open(OUT_TXT, 'w') as f:
        f.write('\n'.join(_lines) + '\n')
    out(f'saved: {OUT_TXT}')


if __name__ == '__main__':
    main()
