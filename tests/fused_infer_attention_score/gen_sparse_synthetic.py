#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic high-sparsity input generator for BlasST validation + analysis.

Why pure-Gaussian Q/K gives ~0% BlasST sparsity, and how to construct
random inputs with controllable high sparsity:

  S = q·k * scale, with q,k ~ N(0,1) elementwise, d=128, scale=1/sqrt(d)
  => S ~ N(0,1) iid over all KV positions, rows, heads.

A KV stack (512 tokens) is skipped iff its rowLoop max is < gm + lambda
(gm = running max from previous stacks).  The max of 512 iid N(0,1) is
~3.3 for *every* stack -- every stack looks identical, so dm ~= 0 > -3
and nothing is ever sparse.  Homogeneously scaling K up does NOT help:
every stack's max scales together with gm.

Sparsity requires *heterogeneity across KV positions*: a few KV tokens
("anchors") with logits far above the bulk.  Real LLM dumps have this
(attention sinks / semantic anchors / peaked distributions), which is why
ChunkedPrefill λ=-3 skips 61% of stacks.

Modes:
  gaussian : q,k ~ N(0,1)                        -> expect ~0% sparse
  scaled   : k ~ N(0, 8^2) homogeneous           -> expect ~0% (scaling no help)
  mixture  : bulk k ~ N(0, 0.5^2) + a few anchor stacks with ramping gains
             -> sparsity controlled by anchor-stack placement
  logit    : q = sqrt(d)*e0, so S = k[...,0]; per-stack logit levels are
             written directly -> deterministic skip pattern incl. a
             boundary-active stack (dm within |lambda| of gm)
  dump     : measure per-stack logit stats on the real ChunkedPrefill dump
             (shows the actual heterogeneity of model data)

Usage:  python gen_sparse_synthetic.py            (CPU only)
"""

import math
import os
import sys

import torch

_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _CUR_DIR)
from blasst_golden_tnd import BlasstGoldenTND  # noqa: E402

Q_LEN = 2048
KV_LEN = 4096
NUM_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 128
LAMBDA = -3.0
KV_STACK = 512


def causal_mask(q_len, device="cpu"):
    """Right-aligned causal int8 mask (1=masked), same format as the dumps."""
    win = torch.ones(q_len, q_len, dtype=torch.int8)
    win = torch.triu(win, diagonal=1)
    return win.to(device)


def measure_sparsity(q, k, v, mask, num_kv_heads):
    golden = BlasstGoldenTND(
        num_heads=NUM_HEADS, num_key_value_heads=num_kv_heads,
        head_dim=HEAD_DIM, scale=1.0 / math.sqrt(HEAD_DIM),
        block_size=32, sparse_lamda=LAMBDA)
    _, _, info = golden.forward_blasst_kernel(
        q, k, v,
        torch.tensor([q.size(0)]), torch.tensor([k.size(0)]),
        masks=[mask])
    return info


def gen_gaussian(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(Q_LEN, NUM_HEADS, HEAD_DIM, generator=g)
    k = torch.randn(KV_LEN, NUM_KV_HEADS, HEAD_DIM, generator=g)
    v = torch.randn(KV_LEN, NUM_KV_HEADS, HEAD_DIM, generator=g)
    return q, k, v


def gen_scaled(k_std=8.0, seed=0):
    q, k, v = gen_gaussian(seed)
    return q, k * k_std, v


def gen_mixture(anchor_stacks=(1, 4), anchor_logits=(16.0, 20.0),
                n_sink_heads=6, bulk_std=0.5, beta=4.0, seed=0,
                q_len=None, kv_len=None, num_heads=None, head_dim=None,
                kv_stack=None, dtype=None, device=None):
    """Mimic the real dump's structure (MHA so each head has its own K):

    - Sink heads: q rows share a common direction u_h (q += beta*u_h) and a
      few KV stacks contain anchor tokens k = gain*u_h.  Every row then gets
      a uniformly high logit at the anchor stacks (like attention sinks /
      vertical lines in real attention maps), so gm jumps for ALL rows and
      the bulk stacks afterwards are skipped at rowLoop granularity.
    - Dense heads: no shared component, no anchors -> stay ~0% sparse
      (like head 3 in the real dump).

    ``anchor_logits`` are the target per-stack logit levels; because the
    shared q component is fixed per row, each row's anchor logit is
    level*(1 + noise*0.5/beta) -- correlated across stacks, so dm margins
    are stable.  Use larger beta (e.g. 8) when anchor levels sit within
    ~2 of a tested sparse_lambda threshold.

    Random-direction anchors do NOT work: per-row anchor logits ~ N(0, g^2),
    ~8% of rows miss a high anchor, and one low-gm row in a 16-row rowLoop
    keeps the whole rowLoop active -> skip rate collapses (measured 17%).

    Shapes default to the module-level demo constants; pass q_len/kv_len/
    num_heads/head_dim/kv_stack/dtype/device to override (e.g. from unit
    tests).
    """
    q_len = q_len or Q_LEN
    kv_len = kv_len or KV_LEN
    num_heads = num_heads or NUM_HEADS
    d = head_dim or HEAD_DIM
    kv_stack = kv_stack or KV_STACK
    g = torch.Generator().manual_seed(seed)
    q = 0.5 * torch.randn(q_len, num_heads, d, generator=g)
    k = bulk_std * torch.randn(kv_len, num_heads, d, generator=g)
    v = torch.randn(kv_len, num_heads, d, generator=g)
    u = torch.randn(num_heads, d, generator=g)
    u = u / u.norm(dim=-1, keepdim=True)
    for h in range(n_sink_heads):
        q[:, h, :] += beta * u[h]
        for stack, logit in zip(anchor_stacks, anchor_logits):
            gain = logit * math.sqrt(d) / beta  # anchor logit = beta*gain/sqrt(d)
            lo = stack * kv_stack
            pos = lo + torch.randperm(kv_stack, generator=g)[:8]
            k[pos, h] = gain * u[h]
    if dtype is not None:
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    if device is not None:
        q, k, v = q.to(device), k.to(device), v.to(device)
    return q, k, v


def gen_logit(stack_levels, noise=0.3, seed=0):
    """q = sqrt(d)*e0 => S = k[...,0].  Write per-stack logit levels
    directly into K channel 0 for a deterministic skip pattern."""
    g = torch.Generator().manual_seed(seed)
    q = torch.zeros(Q_LEN, NUM_HEADS, HEAD_DIM)
    q[:, :, 0] = math.sqrt(HEAD_DIM)
    k = torch.zeros(KV_LEN, NUM_KV_HEADS, HEAD_DIM)
    for s in range(KV_LEN // KV_STACK):
        lvl = stack_levels.get(s, 0.0)
        k[s * KV_STACK:(s + 1) * KV_STACK, :, 0] = \
            lvl + noise * torch.randn(KV_STACK, NUM_KV_HEADS, generator=g)
    v = torch.randn(KV_LEN, NUM_KV_HEADS, HEAD_DIM, generator=g)
    return q, k, v


def dump_logit_stats():
    """Per-stack lm/gm progression on the real ChunkedPrefill dump."""
    from test_fia_dump_cases import load_dump, reconstruct_golden_inputs
    dump = ("/home/z00603376/blasst_model/kvcomp/blasst_res/attention_dumps/"
            "fia_layer30_rank0_stateChunkedPrefill.pt")
    _, inputs, _ = load_dump(dump)
    q, k, v, q_cum, kv_cum, masks = reconstruct_golden_inputs(inputs)
    scale = 1.0 / math.sqrt(q.size(-1))
    n_stacks = KV_LEN // KV_STACK
    print(f"  real dump logits (head-wise, qb=15 full window, {n_stacks} stacks):")
    for h in range(NUM_HEADS):
        s = (q[15 * 128:16 * 128, h].float() @ k[:, 0].float().T) * scale
        win = masks[0][15 * 128:16 * 128]  # (128, 2048) right-aligned window
        add = s.new_zeros(128, KV_LEN)
        add[:, KV_LEN - win.size(1):] = torch.where(
            win != 0, torch.tensor(float("-inf")), torch.tensor(0.0))
        s = s + add
        lm = s.view(128, n_stacks, KV_STACK).max(dim=-1).values  # (128, 8)
        gm = lm[:, 0]
        below = 0
        for st in range(1, n_stacks):
            dm = (lm[:, st] - gm).max()
            if dm.item() < LAMBDA:
                below += 1
            else:
                gm = torch.maximum(gm, lm[:, st])
        print(f"    head {h}: std={s[s.isfinite()].std():.2f} "
              f"max={s.max():.1f} per-stack lm={[f'{x:.1f}' for x in lm.max(dim=0).values.tolist()]} "
              f"stacks_below_gm{LAMBDA:+.0f}={below}/{n_stacks - 1}")
    g = torch.Generator().manual_seed(0)
    sg = (torch.randn(128, HEAD_DIM, generator=g)
          @ torch.randn(KV_LEN, HEAD_DIM, generator=g).T) * scale
    print(f"  gaussian logits: std={sg.std():.2f} max={sg.max():.2f} "
          f"per-stack lm={[f'{x:.1f}' for x in sg.view(128, n_stacks, KV_STACK).max(dim=-1).values.max(dim=0).values.tolist()]}")


def main():
    torch.manual_seed(0)
    mask = causal_mask(Q_LEN)

    print(f"== dump logit heterogeneity (why real data is sparse at λ={LAMBDA}) ==")
    dump_logit_stats()

    print(f"\n== synthetic generators, kernel-granularity golden sparsity "
          f"(q={Q_LEN} kv={KV_LEN} H={NUM_HEADS}, causal) ==")
    cases = [
        ("gaussian  (q,k~N(0,1))", gen_gaussian(), NUM_KV_HEADS, None),
        ("scaled    (k~N(0,8^2)) ", gen_scaled(), NUM_KV_HEADS, None),
        ("mixture   (sink heads 0-5, anchors@stack1,4)",
         gen_mixture(), NUM_HEADS, 336),
        ("logit     (levels 8,0,12,0,11,0,0,0)",
         gen_logit({0: 8.0, 1: 0.0, 2: 12.0, 3: 0.0,
                    4: 11.0, 5: 0.0, 6: 0.0, 7: 0.0}), NUM_KV_HEADS, 448),
    ]
    for name, (q, k, v), hkv, expect in cases:
        info = measure_sparsity(q, k, v, mask, hkv)
        exp = f" expected={expect}" if expect is not None else ""
        print(f"  {name}: sparsity={info['sparsity']:.2%} "
              f"({info['skipped_blocks']}/{info['total_blocks']}){exp}")


if __name__ == "__main__":
    main()
