#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TND-layout BLASST sparse attention golden reference for the migrated
npu_fused_infer_attention_score operator.

This module simulates the NPU rowLoop sparse-detection path on flat TND tensors
(query/key/value shapes are ``(total_tokens, num_heads, head_dim)`` or
``(total_tokens, num_key_value_heads, head_dim)`` for GQA/MQA).  Batch boundaries
are taken from ``actual_seq_lengths`` / ``actual_seq_lengths_kv``.

The sparse decision follows the source warehouse ``BlasstGolden`` logic:
for each query token/head, KV tokens are split into ``block_size`` blocks.
Each block is divided into two half-blocks (rowLoop halves).  A half-block is
marked sparse when ``max(half_block_scores) - global_max < sparse_lamda``.
Only when both halves of a block are sparse is the whole block skipped.
"""

import math
import torch
import torch.nn.functional as F


class BlasstGoldenTND:
    """BLASST golden reference for TND-layout attention.

    Args:
        num_heads: Number of query heads.
        num_key_value_heads: Number of key/value heads (GQA/MQA).
        head_dim: Head dimension.
        scale: Attention score scale factor (typically ``1/sqrt(head_dim)``).
        block_size: KV block size used for sparse detection.  For the
            non-paged TND path this is a logical block size; the kernel uses
            the same concept via its tiling parameters.
        sparse_lamda: BLASST threshold.  ``-99.0`` effectively disables
            sparsity; typical active values are ``-40.0`` (low sparsity) and
            ``-3.0`` (high sparsity).
        rowloop_rows: Rows per rowLoop half-block.  The source implementation
            hard-codes 16, so each ``block_size`` is split into two halves of
            ``rowloop_rows`` rows.  We default to ``block_size // 2`` to match
            the common ``block_size = 32`` / ``rowloop_rows = 16`` setup.
    """

    def __init__(self, num_heads, num_key_value_heads, head_dim, scale,
                 block_size, sparse_lamda, rowloop_rows=None):
        if num_key_value_heads == 0:
            num_key_value_heads = num_heads
        if num_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_key_value_heads ({num_key_value_heads})")
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.group_size = num_heads // num_key_value_heads
        self.head_dim = head_dim
        self.scale = scale
        self.block_size = block_size
        self.sparse_lamda = float(sparse_lamda)
        self.rowloop_rows = rowloop_rows if rowloop_rows is not None else block_size // 2
        if self.rowloop_rows * 2 != block_size:
            raise ValueError(
                f"rowloop_rows*2 ({self.rowloop_rows * 2}) must equal block_size ({block_size})")

    @staticmethod
    def _split_tnd(tensor, actual_seq_lengths):
        """Split a TND tensor into per-batch chunks."""
        chunks = []
        start = 0
        for b in range(actual_seq_lengths.numel()):
            end = int(actual_seq_lengths[b].item())
            chunks.append(tensor[start:end])
            start = end
        return chunks

    def _compute_qk_scores(self, q_token, k_chunk):
        """Compute scores for one query token against one KV chunk.

        ``q_token`` shape: ``(head_dim,)``.
        ``k_chunk`` shape: ``(kv_len, head_dim)`` for the selected KV head.
        Returns ``(kv_len,)`` scores.
        """
        return torch.matmul(q_token.float(), k_chunk.float().T) * self.scale

    def _softmax_attention(self, scores, values):
        """Standard softmax + weighted sum on CPU float32."""
        weights = F.softmax(scores.float(), dim=-1)
        return torch.matmul(weights, values.float())

    def _detect_sparse_block(self, block_scores, global_max):
        """Simulate NPU SP-DETECT for one KV block.

        ``block_scores`` has length ``block_size``; if the real sequence ends
        mid-block we pad with ``-inf`` before calling this function.

        Returns ``(sp1, sp2, sp_res, global_max)`` where ``sp_res`` is True
        only when both half-blocks are sparse.
        """
        scores = block_scores.float()
        row0_scores = scores[:self.rowloop_rows]
        row0_max = row0_scores.max()
        max_diff_0 = row0_max - global_max
        sp1 = 1 if max_diff_0 < self.sparse_lamda else 0
        if sp1 == 0:
            global_max = max(global_max, row0_max)

        row1_scores = scores[self.rowloop_rows:self.block_size]
        row1_max = row1_scores.max()
        max_diff_1 = row1_max - global_max
        sp2 = 1 if max_diff_1 < self.sparse_lamda else 0

        sp_res = sp1 == 1 and sp2 == 1
        return sp1, sp2, sp_res, global_max

    @staticmethod
    def _apply_batch_mask(scores, batch_mask, kv_len):
        """Apply a right-aligned int8 mask (1=masked, 0=visible) to scores.

        ``scores``: ``(..., q_len, kv_len)`` or ``(kv_len,)`` for a single
        query token.  ``batch_mask``: ``(q_len, window)`` with
        ``window <= kv_len``; kv positions before ``kv_len - window`` are
        always visible.  Masked positions are set to ``-inf``.
        Returns the modified scores (a new tensor for the dense path,
        modified in place for the per-token path).
        """
        offset = kv_len - (batch_mask.size(1) if scores.dim() > 1
                           else batch_mask.size(0))
        if scores.dim() == 1:
            qp_mask = batch_mask  # caller passes the single row
            idx = (qp_mask != 0).nonzero().flatten() + offset
            scores[idx] = float('-inf')
            return scores
        q_len = scores.size(-2)
        add = scores.new_zeros(q_len, kv_len)
        window = add[:, offset:]
        window[batch_mask != 0] = float('-inf')
        return scores + add

    def forward_dense(self, query, key, value, actual_seq_lengths,
                      actual_seq_lengths_kv, masks=None):
        """Dense attention golden for TND layout.

        ``masks``: optional list of per-batch masks, each either ``None``
        (fully visible) or an int8 tensor ``(q_len, window)`` right-aligned
        over the KV axis (1=masked, 0=visible).

        Returns ``(attention_out, softmax_lse)`` with shapes matching the NPU
        operator: ``attention_out`` is ``(total_q_tokens, num_heads, head_dim)``
        and ``softmax_lse`` is ``(total_q_tokens, num_heads, 1)``.
        """
        out_list, lse_list = [], []
        q_batches = self._split_tnd(query, actual_seq_lengths)
        k_batches = self._split_tnd(key, actual_seq_lengths_kv)
        v_batches = self._split_tnd(value, actual_seq_lengths_kv)

        for b_idx, (q, k, v) in enumerate(zip(q_batches, k_batches, v_batches)):
            batch_mask = masks[b_idx] if masks else None
            # For GQA/MQA repeat KV heads to match the number of query heads.
            if k.size(1) != self.num_heads:
                k = k.repeat_interleave(self.group_size, dim=1)
                v = v.repeat_interleave(self.group_size, dim=1)
            # scores: (num_heads, q_len, kv_len)
            scores = torch.einsum('qhd,khd->hqk', q, k) * self.scale
            if batch_mask is not None:
                scores = self._apply_batch_mask(scores, batch_mask, k.size(0))
            lse = torch.logsumexp(scores, dim=-1, keepdim=True)
            attn = F.softmax(scores, dim=-1)
            out = torch.einsum('hqk,khd->qhd', attn, v)
            out_list.append(out)
            lse_list.append(lse.permute(1, 0, 2))

        return torch.cat(out_list, dim=0), torch.cat(lse_list, dim=0)

    def forward_blasst(self, query, key, value, actual_seq_lengths,
                       actual_seq_lengths_kv, masks=None):
        """BLASST sparse attention golden for TND layout.

        ``masks``: optional list of per-batch masks, each either ``None``
        (fully visible) or an int8 tensor ``(q_len, window)`` right-aligned
        over the KV axis (1=masked, 0=visible).  Masked positions are
        excluded from both sparse detection and the softmax.

        Returns ``(attention_out, softmax_lse, info)`` where ``info`` reports
        how many KV blocks were evaluated/skipped.
        """
        total_blocks = 0
        skipped_blocks = 0
        out_list, lse_list = [], []

        q_batches = self._split_tnd(query, actual_seq_lengths)
        k_batches = self._split_tnd(key, actual_seq_lengths_kv)
        v_batches = self._split_tnd(value, actual_seq_lengths_kv)

        for b_idx, (q, k, v) in enumerate(zip(q_batches, k_batches, v_batches)):
            batch_mask = masks[b_idx] if masks else None
            q_len = q.size(0)
            kv_len = k.size(0)
            num_blocks = (kv_len + self.block_size - 1) // self.block_size

            batch_out = []
            batch_lse = []
            for qh in range(self.num_heads):
                kvh = qh // self.group_size
                k_head = k[:, kvh, :]
                v_head = v[:, kvh, :]

                head_out = []
                head_lse = []
                for qp in range(q_len):
                    qt = q[qp, qh]
                    all_scores = self._compute_qk_scores(qt, k_head)
                    if batch_mask is not None:
                        all_scores = self._apply_batch_mask(
                            all_scores, batch_mask[qp], kv_len)

                    # Match the kernel's accounting: blocks entirely inside the
                    # masked-out region are never visited by the KV loop
                    # (kvSLoopNumTotal is truncated at the visible boundary),
                    # so they are excluded from total/skipped block counts.
                    if batch_mask is not None:
                        offset = kv_len - batch_mask.size(1)
                        visible_len = offset + int((batch_mask[qp] == 0).sum().item())
                        num_counted_blocks = (visible_len + self.block_size - 1) \
                            // self.block_size
                    else:
                        num_counted_blocks = num_blocks

                    global_max = float('-inf')
                    active_scores = []
                    active_values = []

                    for blk in range(num_counted_blocks):
                        total_blocks += 1
                        start = blk * self.block_size
                        end = min(start + self.block_size, kv_len)
                        blk_scores = all_scores[start:end]

                        if blk_scores.numel() < self.block_size:
                            pad_len = self.block_size - blk_scores.numel()
                            blk_scores = torch.cat([
                                blk_scores,
                                torch.full((pad_len,), float('-inf'),
                                           dtype=blk_scores.dtype,
                                           device=blk_scores.device)
                            ])

                        _, _, sp_res, global_max = self._detect_sparse_block(
                            blk_scores, global_max)
                        if sp_res:
                            skipped_blocks += 1
                            continue

                        active_scores.append(all_scores[start:end])
                        active_values.append(v_head[start:end])

                    if active_scores:
                        merged_s = torch.cat(active_scores, dim=0)
                        merged_v = torch.cat(active_values, dim=0)
                        head_out.append(self._softmax_attention(merged_s, merged_v))
                        head_lse.append(torch.logsumexp(merged_s, dim=0))
                    else:
                        head_out.append(torch.zeros(self.head_dim,
                                                    dtype=torch.float32,
                                                    device=q.device))
                        head_lse.append(torch.tensor(float('-inf'),
                                                     dtype=torch.float32,
                                                     device=q.device))

                batch_out.append(torch.stack(head_out, dim=0))
                batch_lse.append(torch.stack(head_lse, dim=0))

            # batch_out: (num_heads, q_len, head_dim) -> (q_len, num_heads, head_dim)
            out_list.append(torch.stack(batch_out, dim=0).permute(1, 0, 2))
            # batch_lse: (num_heads, q_len) -> (q_len, num_heads, 1)
            lse_list.append(torch.stack(batch_lse, dim=0).permute(1, 0).unsqueeze(-1))

        info = {
            "total_blocks": total_blocks,
            "skipped_blocks": skipped_blocks,
            "sparsity": skipped_blocks / max(total_blocks, 1),
        }
        return torch.cat(out_list, dim=0), torch.cat(lse_list, dim=0), info

    def _kernel_skip_decisions(self, scores, no_skip_kv, q_sblock_size,
                               kv_stack, rowloop_rows):
        """Simulate the kernel's per-stack sparse-skip decisions for one
        (qSBlock, q head) tile.

        ``scores``: ``(rows, no_skip_kv)`` float32, mask already applied
        (masked positions are ``-inf``).  ``rows <= q_sblock_size``.

        Returns ``(group_active, total_stacks, skipped_stacks)`` where
        ``group_active`` is a list of length ``num_stacks``; each entry is a
        list of bools, one per ``rowloop_rows``-row group within the tile,
        marking whether that row group receives the stack's KV.

        Kernel semantics replicated (flash_attention_regular.h +
        block_epilogue_online_softmax.hpp + block_mmad_pv.hpp +
        block_epilogue_rescale_o.hpp):
          * KV is processed in stacks of ``kv_stack`` (MAX_KV_STACK_LEN=512).
          * The q tile is split into 2 vector subBlocks of ``rows//2``.
          * Each subBlock is processed in rowLoops of ``rowloop_rows`` (16).
          * Per rowLoop: ``dm = max_over_rows(lm - gm)``; sparse iff
            ``dm < sparse_lamda``.  Detection within a subBlock stops at the
            first non-sparse rowLoop, so sparse rowLoops always form a
            prefix of the subBlock.
          * Sparse rowLoops skip the stack for their 16 rows: P is never
            computed, the RescaleO update zeroes their PV contribution
            (``sp_flag_row_offset`` prefix) and keeps their accumulator,
            row-sum and ``gm`` unchanged.
          * The first stack is never sparse (``gm = lm``).
          * A stack is skipped at the PV level (and counted in
            ``blockSparseCount``) iff BOTH subBlocks are fully sparse.
          * ``gm`` per row: max of ``lm`` over rowLoops that were processed
            non-sparse; rows in sparse rowLoops keep their stale ``gm``.
        """
        rows = scores.size(0)
        num_stacks = (no_skip_kv + kv_stack - 1) // kv_stack
        gm = torch.full((rows,), float('-inf'), dtype=torch.float32,
                        device=scores.device)
        num_groups = (rows + rowloop_rows - 1) // rowloop_rows
        group_active = []
        skipped = 0
        sub0_end = rows // 2

        for s in range(num_stacks):
            kv_start = s * kv_stack
            kv_end = min(kv_start + kv_stack, no_skip_kv)
            lm = scores[:, kv_start:kv_end].max(dim=-1).values

            if s == 0:
                # First stack: never sparse, gm initialized from lm.
                gm = lm.clone()
                group_active.append([True] * num_groups)
                continue

            stack_active = [False] * num_groups
            sub_flags = []
            for sub_start, sub_end in ((0, sub0_end), (sub0_end, rows)):
                # Kernel: sp_flag = 1 only when the subBlock has rows.
                flag = 1 if sub_end > sub_start else 0
                rl = sub_start
                while rl < sub_end:
                    rl_end = min(rl + rowloop_rows, sub_end)
                    grp = rl // rowloop_rows
                    if flag == 1:
                        dm = (lm[rl:rl_end] - gm[rl:rl_end]).max()
                        if bool(dm < self.sparse_lamda):
                            # Sparse rowLoop: rows skip this stack; gm, row
                            # sum and accumulator stay unchanged.
                            rl = rl_end
                            continue
                        flag = 0
                    stack_active[grp] = True
                    gm[rl:rl_end] = torch.maximum(gm[rl:rl_end], lm[rl:rl_end])
                    rl = rl_end
                sub_flags.append(flag)

            # PV-level skip: both subBlock flags must be 1.  An empty
            # subBlock has flag=0, so it can never trigger a skip.
            if sub_flags[0] == 1 and sub_flags[1] == 1:
                skipped += 1
            group_active.append(stack_active)

        return group_active, num_stacks, skipped

    def forward_blasst_kernel(self, query, key, value, actual_seq_lengths,
                              actual_seq_lengths_kv, masks=None,
                              q_sblock=128, kv_stack=512, rowloop_rows=16):
        """BLASST sparse attention golden with kernel-exact skip granularity.

        Skip unit: (qSBlock of ``q_sblock`` q tokens) x (one q head) x
        (KV stack of ``kv_stack``), matching the NPU regular kernel.
        Attention output per row is the exact softmax over the active
        (non-skipped) KV stacks; masked positions are excluded from both
        the skip detection and the softmax.

        Returns ``(attention_out, softmax_lse, info)``.
        """
        total_blocks = 0
        skipped_blocks = 0
        out_list, lse_list = [], []

        q_batches = self._split_tnd(query, actual_seq_lengths)
        k_batches = self._split_tnd(key, actual_seq_lengths_kv)
        v_batches = self._split_tnd(value, actual_seq_lengths_kv)

        for b_idx, (q, k, v) in enumerate(zip(q_batches, k_batches, v_batches)):
            batch_mask = masks[b_idx] if masks else None
            q_len = q.size(0)
            kv_len = k.size(0)
            diff_s = max(0, kv_len - q_len)

            batch_out = torch.zeros(q_len, self.num_heads, self.head_dim,
                                    dtype=torch.float32, device=q.device)
            batch_lse = torch.full((q_len, self.num_heads), float('-inf'),
                                   dtype=torch.float32, device=q.device)

            for qh in range(self.num_heads):
                kvh = qh // self.group_size
                k_head = k[:, kvh, :].float()
                v_head = v[:, kvh, :].float()

                for qb in range((q_len + q_sblock - 1) // q_sblock):
                    q_start = qb * q_sblock
                    q_end = min(q_start + q_sblock, q_len)
                    rows = q_end - q_start

                    # Kernel (flash_attention_regular.h:548-611): the causal
                    # truncation noSkipKvS = min(kv_len, (qb+1)*q_sblock +
                    # diff_s) only applies when a mask is present; without a
                    # mask the KV loop always covers the full kv_len.
                    if batch_mask is not None:
                        no_skip = min(kv_len, q_end + diff_s)
                    else:
                        no_skip = kv_len

                    # Full visible-range scores for this tile: (rows, no_skip)
                    scores = torch.matmul(
                        q[q_start:q_end, qh, :].float(), k_head[:no_skip].T
                    ) * self.scale
                    if batch_mask is not None:
                        offset = kv_len - batch_mask.size(1)
                        add = scores.new_zeros(rows, no_skip)
                        win_start = max(offset, 0)
                        if no_skip > win_start:
                            win = add[:, win_start:no_skip]
                            mask_slice = batch_mask[q_start:q_end,
                                                    :no_skip - win_start]
                            win[mask_slice != 0] = float('-inf')
                        scores = scores + add

                    group_active, n_stacks, n_skipped = \
                        self._kernel_skip_decisions(
                            scores, no_skip, q_sblock, kv_stack, rowloop_rows)
                    total_blocks += n_stacks
                    skipped_blocks += n_skipped

                    # Per rowLoop group: exact softmax over the KV of stacks
                    # where the group was not sparse-skipped.
                    num_groups = (rows + rowloop_rows - 1) // rowloop_rows
                    for grp in range(num_groups):
                        g_start = grp * rowloop_rows
                        g_end = min(g_start + rowloop_rows, rows)
                        col_idx = []
                        for s in range(n_stacks):
                            if group_active[s][grp]:
                                kv_start = s * kv_stack
                                kv_end = min(kv_start + kv_stack, no_skip)
                                col_idx.append(torch.arange(
                                    kv_start, kv_end, device=scores.device))
                        if not col_idx:
                            batch_out[q_start + g_start:q_start + g_end,
                                      qh, :] = 0.0
                            batch_lse[q_start + g_start:q_start + g_end,
                                      qh] = float('-inf')
                            continue
                        cols = torch.cat(col_idx)
                        act_scores = scores[g_start:g_end][:, cols]
                        batch_lse[q_start + g_start:q_start + g_end, qh] = \
                            torch.logsumexp(act_scores, dim=-1)
                        weights = F.softmax(act_scores, dim=-1)
                        batch_out[q_start + g_start:q_start + g_end, qh, :] = \
                            torch.matmul(weights, v_head[cols])

            out_list.append(batch_out)
            lse_list.append(batch_lse.unsqueeze(-1))

        info = {
            "total_blocks": total_blocks,
            "skipped_blocks": skipped_blocks,
            "sparsity": skipped_blocks / max(total_blocks, 1),
        }
        return torch.cat(out_list, dim=0), torch.cat(lse_list, dim=0), info

    @staticmethod
    def compute_metrics(out_test, out_ref):
        """Compute max/mean/relative error and cosine similarity."""
        out_t = out_test.float().flatten()
        out_r = out_ref.float().flatten()
        diff = (out_t - out_r).abs()
        cos_sim = F.cosine_similarity(out_t, out_r, dim=0).item()
        rel_err = (torch.norm(out_t - out_r) / (torch.norm(out_r) + 1e-9)).item()
        return {
            "max_err": diff.max().item(),
            "mean_err": diff.mean().item(),
            "rel_err": rel_err,
            "cos_sim": cos_sim,
        }


def build_from_tnd_inputs(query, key, value, actual_seq_lengths,
                          actual_seq_lengths_kv, num_heads,
                          num_key_value_heads, scale, block_size,
                          sparse_lamda):
    """Convenience factory to build a ``BlasstGoldenTND`` from TND tensors."""
    if num_key_value_heads == 0:
        num_key_value_heads = num_heads
    head_dim = query.size(-1)
    return BlasstGoldenTND(
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        scale=scale,
        block_size=block_size,
        sparse_lamda=sparse_lamda,
    )
