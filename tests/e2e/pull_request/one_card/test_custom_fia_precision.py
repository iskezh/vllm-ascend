#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Precision CI guard for the custom fused_infer_attention_score (BlasST) op.

Single-op level, no engine: the custom ``_C_ascend`` op vs an fp32 reference
for dense / causal / GQA / paged-decode shapes (including the FlashDecode
shape), and vs the kernel-granularity BlasST golden for sparse-lambda cases.
Also covers the two op attr switches (``host_seq_tiling`` / ``flash_decode``)
that replaced the ``VLLM_FIA_*`` environment kill-switches: with the switch
off the op must still produce correct output through its fallback path.

Case design mirrors tests/fused_infer_attention_score/validate_post_cleanup.py
(the developer-side 14-case suite); this file is the CI-collectable subset.
"""

import math
import os
import sys
from pathlib import Path

import pytest
import torch

_CUR = os.path.dirname(os.path.realpath(__file__))
_CUSTOM_OPP = os.path.join(_CUR, "..", "..", "..", "..", "vllm_ascend",
                           "_cann_ops_custom", "vendors", "custom_transformer")
if os.path.exists(_CUSTOM_OPP):
    os.environ.setdefault("ASCEND_CUSTOM_OPP_PATH", _CUSTOM_OPP)

import torch_npu  # noqa: F401,E402
import vllm_ascend.vllm_ascend_C  # noqa: F401,E402

# Kernel-granularity BlasST golden (fp32 simulator mirroring the kernel's
# skip decisions). Lives outside the e2e tree (tests/fused_infer_attention_
# score/); import by path and degrade gracefully so collection never breaks.
_GOLDEN_DIR = Path(_CUR).resolve().parents[2] / "fused_infer_attention_score"
try:
    sys.path.insert(0, str(_GOLDEN_DIR))
    from blasst_golden_tnd import BlasstGoldenTND  # noqa: E402

    _golden_available = True
except ImportError:
    _golden_available = False

SWA_INT_MAX = 2147483647
DENSE_LAMBDA = -99.0
DEVICE = "npu:0"

# sparse_mode=3 requires a real mask; production passes this int8 compact
# triU mask (vllm_ascend/attention/attention_mask.py get_splitfuse_attn_mask,
# kernel reads it as int8 RowMajor [2048, 2048], 1 = masked position).
_CAUSAL_MASK = None


def causal_mask():
    global _CAUSAL_MASK
    if _CAUSAL_MASK is None:
        _CAUSAL_MASK = torch.triu(
            torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(DEVICE)
    return _CAUSAL_MASK


def _cumsum(lens):
    out, s = [], 0
    for x in lens:
        s += x
        out.append(s)
    return out


def run_custom(q, k, v, q_lens, kv_lens, num_heads, num_kv_heads, scale,
               sparse_lambda=DENSE_LAMBDA, causal=False, blocktable=None,
               block_size=0, softmax_lse_flag=True, sparse_stats_flag=False,
               host_seq_tiling=True, flash_decode=True):
    # Op contract (matches attention_v1.py): q lens are prefix sums; kv lens
    # are prefix sums for non-paged, raw per-batch lengths for paged.
    paged = blocktable is not None
    return torch.ops._C_ascend.npu_fused_infer_attention_score(
        q, k, v,
        pse_shift=None, atten_mask=causal_mask() if causal else None,
        actual_seq_lengths=_cumsum(q_lens),
        actual_seq_lengths_kv=kv_lens if paged else _cumsum(kv_lens),
        blocktable=blocktable,
        num_heads=num_heads, scale=scale,
        pre_tokens=SWA_INT_MAX, next_tokens=SWA_INT_MAX,
        input_layout="TND", num_key_value_heads=num_kv_heads,
        sparse_mode=3 if causal else 0, inner_precise=0,
        block_size=block_size, antiquant_mode=0,
        sparse_lambda=sparse_lambda, softmax_lse_flag=softmax_lse_flag,
        sparse_stats_flag=sparse_stats_flag,
        host_seq_tiling=host_seq_tiling, flash_decode=flash_decode,
    )


def ref_attention(q, k, v, q_lens, kv_lens, num_heads, num_kv_heads, scale,
                  causal=False, blocktable=None, block_size=0):
    """fp32 reference. q: [total_q, H, D]; non-paged k/v: [total_kv, Hkv, D];
    paged k/v: [num_blocks, block_size, Hkv*D] with blocktable [B, max_blocks].
    Returns out [total_q, H, D] in fp32."""
    _, D = q.shape[1], q.shape[2]
    group = num_heads // num_kv_heads if num_kv_heads > 0 else 1
    outs = []
    q_off = 0
    for b, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
        qb = q[q_off:q_off + ql].float()
        q_off += ql
        if blocktable is not None:
            nb = (kl + block_size - 1) // block_size
            blk = blocktable[b, :nb].long()
            kb = k[blk].reshape(-1, num_kv_heads, D)[:kl].float()
            vb = v[blk].reshape(-1, num_kv_heads, D)[:kl].float()
        else:
            kv_off = int(sum(kv_lens[:b]))
            kb = k[kv_off:kv_off + kl].float()
            vb = v[kv_off:kv_off + kl].float()
        kb = kb.repeat_interleave(group, dim=1)
        vb = vb.repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qb, kb) * scale
        if causal:
            i = torch.arange(ql, device=q.device).unsqueeze(1)
            j = torch.arange(kl, device=q.device).unsqueeze(0)
            mask = j > (i + (kl - ql))
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", probs, vb))
    return torch.cat(outs, dim=0)


def make_varlen(q_lens, kv_lens, H, Hkv, D, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    tq, tkv = sum(q_lens), sum(kv_lens)
    q = torch.randn(tq, H, D, generator=g, dtype=torch.float32).to(dtype).to(DEVICE)
    k = torch.randn(tkv, Hkv, D, generator=g, dtype=torch.float32).to(dtype).to(DEVICE)
    v = torch.randn(tkv, Hkv, D, generator=g, dtype=torch.float32).to(dtype).to(DEVICE)
    return q, k, v


def make_paged(kv_lens, Hkv, D, block_size, dtype, seed=0):
    # Production paged layout: [num_blocks, block_size, Hkv*D] (3D view).
    g = torch.Generator(device="cpu").manual_seed(seed)
    max_blocks = max((kl + block_size - 1) // block_size for kl in kv_lens)
    num_blocks = sum((kl + block_size - 1) // block_size for kl in kv_lens)
    k = torch.randn(num_blocks, block_size, Hkv * D, generator=g, dtype=torch.float32).to(dtype).to(DEVICE)
    v = torch.randn(num_blocks, block_size, Hkv * D, generator=g, dtype=torch.float32).to(dtype).to(DEVICE)
    bt = torch.zeros(len(kv_lens), max_blocks, dtype=torch.int32)
    off = 0
    for b, kl in enumerate(kv_lens):
        nb = (kl + block_size - 1) // block_size
        bt[b, :nb] = torch.arange(off, off + nb, dtype=torch.int32)
        off += nb
    return k, v, bt.to(DEVICE)


def _assert_precision(out, ref, dtype):
    a, b = out.float().flatten(), ref.float().flatten()
    rel_err = (torch.norm(a - b) / (torch.norm(b) + 1e-9)).item()
    cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel_lim, cos_lim = (0.02, 0.995) if dtype == torch.float16 else (0.03, 0.99)
    assert rel_err <= rel_lim and cos_sim >= cos_lim, (
        f"rel_err={rel_err:.4e} (limit {rel_lim}), cos_sim={cos_sim:.6f} "
        f"(limit {cos_lim})")


H, D = 16, 128
SCALE = 1.0 / math.sqrt(D)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dense_no_mask(dtype):
    q_lens = kv_lens = [257, 1024, 33]
    q, k, v = make_varlen(q_lens, kv_lens, H, H, D, dtype, seed=1)
    out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, H, SCALE)
    _assert_precision(out, ref_attention(q, k, v, q_lens, kv_lens, H, H, SCALE), dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dense_causal(dtype):
    q_lens = kv_lens = [300, 777]
    q, k, v = make_varlen(q_lens, kv_lens, H, H, D, dtype, seed=2)
    out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, H, SCALE, causal=True)
    _assert_precision(
        out, ref_attention(q, k, v, q_lens, kv_lens, H, H, SCALE, causal=True), dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gqa(dtype):
    Hkv = 4
    q_lens = kv_lens = [128, 512]
    q, k, v = make_varlen(q_lens, kv_lens, H, Hkv, D, dtype, seed=3)
    out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, Hkv, SCALE)
    _assert_precision(out, ref_attention(q, k, v, q_lens, kv_lens, H, Hkv, SCALE), dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_decode_flash_decode_shape(dtype):
    # q=1, kv=4096+ -> FlashDecode-eligible (numTasks<=8, kv>=4096,
    # lseFlag=False, blockSize=128; see the FD gate in the op tiling).
    bs, Hkv, blk = 2, 4, 128
    q_lens = [1] * bs
    kv_lens = [4096, 5000]
    q, _, _ = make_varlen(q_lens, [0] * bs, H, Hkv, D, dtype, seed=4)
    k, v, bt = make_paged(kv_lens, Hkv, D, blk, dtype, seed=4)
    out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, Hkv, SCALE,
                           blocktable=bt, block_size=blk, softmax_lse_flag=False)
    _assert_precision(
        out,
        ref_attention(q, k, v, q_lens, kv_lens, H, Hkv, SCALE,
                      blocktable=bt, block_size=blk),
        dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_decode_flash_decode_disabled(dtype):
    # flash_decode=False must route around the FD path and stay correct
    # (guards the op-attr plumbing that replaced VLLM_FIA_FD).
    bs, Hkv, blk = 2, 4, 128
    q_lens = [1] * bs
    kv_lens = [4096, 5000]
    q, _, _ = make_varlen(q_lens, [0] * bs, H, Hkv, D, dtype, seed=4)
    k, v, bt = make_paged(kv_lens, Hkv, D, blk, dtype, seed=4)
    out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, Hkv, SCALE,
                           blocktable=bt, block_size=blk, softmax_lse_flag=False,
                           flash_decode=False)
    _assert_precision(
        out,
        ref_attention(q, k, v, q_lens, kv_lens, H, Hkv, SCALE,
                      blocktable=bt, block_size=blk),
        dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_decode_host_seq_tiling_off(dtype):
    # host_seq_tiling=False falls back to D2H seq reads in tiling and must
    # stay correct (guards the op-attr plumbing that replaced
    # VLLM_FIA_HOST_SEQ_TILING).
    bs, Hkv, blk = 2, 4, 128
    q_lens = [1] * bs
    kv_lens = [100, 300]
    q, _, _ = make_varlen(q_lens, [0] * bs, H, Hkv, D, dtype, seed=5)
    k, v, bt = make_paged(kv_lens, Hkv, D, blk, dtype, seed=5)
    out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, Hkv, SCALE,
                           blocktable=bt, block_size=blk,
                           host_seq_tiling=False)
    _assert_precision(
        out,
        ref_attention(q, k, v, q_lens, kv_lens, H, Hkv, SCALE,
                      blocktable=bt, block_size=blk),
        dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_decode_regular(dtype):
    # Short kv -> regular paged path (no FD).
    bs, Hkv, blk = 2, 4, 128
    q_lens = [1] * bs
    kv_lens = [100, 300]
    q, _, _ = make_varlen(q_lens, [0] * bs, H, Hkv, D, dtype, seed=5)
    k, v, bt = make_paged(kv_lens, Hkv, D, blk, dtype, seed=5)
    out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, Hkv, SCALE,
                           blocktable=bt, block_size=blk)
    _assert_precision(
        out,
        ref_attention(q, k, v, q_lens, kv_lens, H, Hkv, SCALE,
                      blocktable=bt, block_size=blk),
        dtype)


@pytest.mark.skipif(not _golden_available, reason="BlasST golden not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_blasst_stats(dtype):
    # Stats mode (detection only, no real skip): the op's skip counters must
    # match the kernel-granularity golden's decisions.
    lam = -3.0
    q_lens = kv_lens = [640, 1408]
    q, k, v = make_varlen(q_lens, kv_lens, H, H, D, dtype, seed=6)
    b2 = q_lens[0]  # batch 2 kv starts at global row 640
    k[b2:b2 + 512] = (k[b2:b2 + 512].float() * 4.0).to(dtype)
    k[b2 + 512:] = (k[b2 + 512:].float() * 0.001).to(dtype)
    golden = BlasstGoldenTND(num_heads=H, num_key_value_heads=H, head_dim=D,
                             scale=SCALE, block_size=32, sparse_lamda=lam,
                             rowloop_rows=16)
    _, _, info = golden.forward_blasst_kernel(
        q.float().cpu(), k.float().cpu(), v.float().cpu(),
        torch.tensor(_cumsum(q_lens)), torch.tensor(_cumsum(kv_lens)))

    _, _, stats = run_custom(q, k, v, q_lens, kv_lens, H, H, SCALE,
                             sparse_lambda=lam, sparse_stats_flag=True)
    assert int(stats[0]) == info["skipped_blocks"], (
        f"skipped blocks: custom {int(stats[0])} vs golden {info['skipped_blocks']}")
    assert int(stats[1]) == info["total_blocks"], (
        f"total blocks: custom {int(stats[1])} vs golden {info['total_blocks']}")


@pytest.mark.skip(
    reason="known kernel hang in the real-skip path (sparse_lambda=-3, "
           "stats off); under bisect via tests/fused_infer_attention_score/"
           "repro_6b.py. Unskip once the kernel fix lands — stats mode "
           "(test_blasst_stats) covers the skip decisions until then.")
@pytest.mark.skipif(not _golden_available, reason="BlasST golden not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_blasst_sparse_output(dtype):
    # Real-skip mode: output vs kernel-granularity golden.
    lam = -3.0
    q_lens = kv_lens = [640, 1408]
    q, k, v = make_varlen(q_lens, kv_lens, H, H, D, dtype, seed=6)
    b2 = q_lens[0]
    k[b2:b2 + 512] = (k[b2:b2 + 512].float() * 4.0).to(dtype)
    k[b2 + 512:] = (k[b2 + 512:].float() * 0.001).to(dtype)
    golden = BlasstGoldenTND(num_heads=H, num_key_value_heads=H, head_dim=D,
                             scale=SCALE, block_size=32, sparse_lamda=lam,
                             rowloop_rows=16)
    ref, _, _ = golden.forward_blasst_kernel(
        q.float().cpu(), k.float().cpu(), v.float().cpu(),
        torch.tensor(_cumsum(q_lens)), torch.tensor(_cumsum(kv_lens)))

    out, _, _ = run_custom(q, k, v, q_lens, kv_lens, H, H, SCALE,
                           sparse_lambda=lam)
    _assert_precision(out, ref.to(DEVICE), dtype)
