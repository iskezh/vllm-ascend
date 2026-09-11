#!/usr/bin/env python3
"""Read the custom FIA op's own skip-ratio return (sparse_stats output).

stats[0] = skipped KV blocks, stats[1] = total KV blocks (kernel-aggregated),
gated by sparse_stats_flag=True. Covers Qwen3.5 shapes (D=256, GQA 6:1 ->
probe 12:2), decode (q=1) and prefill (q=kv) query modes, several KV lengths,
lambda -3 / -1, and two data distributions:

  iid       - independent gaussian K/V (upper-bracket: scores diffuse)
  clustered - K/V built from topic centroids; query from topic 0
              (lower-bracket: off-topic blocks are clearly skippable)

Real-text K/V lives between these brackets.
"""
import math
import sys

sys.path.insert(0, "/home/z00603376/fia/vllm-ascend/tests/fused_infer_attention_score")
import torch  # noqa: E402
from validate_post_cleanup import DEVICE, run_custom  # noqa: E402

torch.manual_seed(7)
D = 256
H, HKV = 12, 2
SCALE = 1.0 / math.sqrt(D)
DT = torch.float16
KV_LENS = [2048, 8192, 28672, 65536]
LAMBDAS = [-3.0, -1.0]


def gen(mode, kv_len, q_len, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if mode == "iid":
        k = torch.randn(kv_len, HKV, D, generator=g)
        v = torch.randn(kv_len, HKV, D, generator=g)
        q = torch.randn(q_len, H, D, generator=g)
    else:  # clustered: block topic every 128 tokens (kernel block size)
        n_topics = 4
        topics = torch.randn(n_topics, 1, D, generator=g)
        topics = topics / topics.norm(dim=-1, keepdim=True)
        nblk = (kv_len + 127) // 128
        assign = torch.arange(nblk) % n_topics
        base = topics[assign].repeat_interleave(128, dim=0)[:kv_len]  # [kv,1,D]
        k = base + 0.05 * torch.randn(kv_len, 1, D, generator=g)
        k = k.repeat(1, HKV, 1)
        v = torch.randn(kv_len, HKV, D, generator=g)
        q = (topics[0] + 0.05 * torch.randn(1, D, generator=g)).repeat(q_len, 1, 1).repeat(1, H // 1, 1)[:, :H, :]
        q = q[:, :H, :]
    return q.to(DT).to(DEVICE), k.to(DT).to(DEVICE), v.to(DT).to(DEVICE)


def probe(mode, kv_len, qmode, lam):
    q_len = 1 if qmode == "decode" else kv_len
    q, k, v = gen(mode, kv_len, q_len, seed=kv_len // 128 + int(-lam) * 100)
    q_lens = [q_len]
    kv_lens = [kv_len]
    try:
        _, _, stats = run_custom(q, k, v, q_lens, kv_lens, H, HKV, SCALE,
                                 sparse_lambda=lam, causal=(qmode == "prefill"),
                                 sparse_stats_flag=True)
        skip, tot = int(stats[0]), int(stats[1])
        return f"{skip:>7}/{tot:<7} {skip / max(tot, 1):>6.1%}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {str(e)[:60]}"


print(f"{'data':<10}{'kv_len':>7}{'q':>8}{'lam':>5}  skip/total ratio")
for mode in ("iid", "clustered"):
    for kv_len in KV_LENS:
        for qmode in ("decode", "prefill"):
            for lam in LAMBDAS:
                r = probe(mode, kv_len, qmode, lam)
                print(f"{mode:<10}{kv_len:>7}{qmode:>8}{lam:>5}  {r}", flush=True)
