#!/usr/bin/env python3
"""Full per-block dm = (local_max - running_global_max) distribution on REAL
dumped K/V, exactly simulating the kernel's sp decision walk.

For each layer, each q head, each KV block (128), walk in order, maintain a
running global max, and record dm = block_half_max - gmax for every half-block
(64) the way the kernel's SP-DETECT does. Reports the full distribution so we
can see where a threshold would actually start skipping.
"""
import math
import sys

sys.path.insert(0, "/home/z00603376/fia/vllm-ascend/tests/fused_infer_attention_score")
import torch  # noqa: E402

SCALE = 1.0 / math.sqrt(256)
DUMP = sys.argv[1] if len(sys.argv) > 1 else "/home/z00603376/longbench_kv_dump.pt"
N_H, N_KV = 24, 4
BLK, HALF = 128, 64


def collect(qmode):
    d = torch.load(DUMP, map_location="cpu")
    out = {}
    for L in sorted(d["layers"]):
        if qmode == "decode":
            q = d["layers"][L]["q"][-1:].float()   # [1,24,256]
            q_lens = 1
        else:
            q = d["layers"][L]["q"].float()        # [T,24,256]
            q_lens = q.size(0)
        k = d["layers"][L]["k"].float()            # [T,4,256]
        s = torch.einsum("qhd,tkd->hkt", q, k) * SCALE   # [24,4,T] (decode q=1)
        T = s.size(-1)
        dm_all = []
        for h in range(N_H):
            kvh = h // (N_H // N_KV)
            row = s[h, kvh]                        # [T]
            gmax = float("-inf")
            nb = (T + BLK - 1) // BLK
            for b in range(nb):
                seg = row[b * BLK:(b + 1) * BLK]
                for half in (seg[:HALF], seg[HALF:]):
                    if half.numel() == 0:
                        continue
                    hm = half.max().item()
                    dm_all.append(hm - gmax)
                    gmax = max(gmax, hm)
        out[L] = torch.tensor(dm_all)
    return out


def report(tag, res):
    print(f"\n===== {tag} =====")
    for L, dm in res.items():
        dmn = dm.numpy() if hasattr(dm, "numpy") else dm
        import numpy as np
        a = np.asarray(dmn)
        qs = np.percentile(a, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        print(f"L{L}: n={len(a)} half-blocks")
        print(f"   dm range [{a.min():.3f}, {a.max():.3f}]  mean={a.mean():.3f}")
        print(f"   percentiles: p0={qs[0]:.3f} p1={qs[1]:.3f} p5={qs[2]:.3f} "
              f"p25={qs[3]:.3f} p50={qs[4]:.3f} p75={qs[5]:.3f} p95={qs[6]:.3f} p99={qs[7]:.3f} p100={qs[8]:.3f}")
        for th in (-3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.3, -0.2, -0.1):
            frac = (a < th).mean()
            print(f"     dm < {th:>5}: {frac:7.4%}  ({int((a < th).sum())} halves)")


if __name__ == "__main__":
    report("DECODE (q = last token)", collect("decode"))
