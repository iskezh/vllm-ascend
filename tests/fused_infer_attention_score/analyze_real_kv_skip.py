#!/usr/bin/env python3
"""Run the custom FIA op on REAL dumped K/V and read skip ratios.

Loads real_kv_dump.pt (layers 3/31/63, real prompt), feeds each layer's
Q/K/V through the op with sparse_stats_flag=True at multiple lambdas, both
decode (q=last token) and chunked-prefill (q=kv) modes.
"""
import math
import sys

sys.path.insert(0, "/home/z00603376/fia/vllm-ascend/tests/fused_infer_attention_score")
import torch  # noqa: E402
from validate_post_cleanup import DEVICE, run_custom, _cumsum  # noqa: E402

D = 256
SCALE = 1.0 / math.sqrt(D)
DUMP = sys.argv[1] if len(sys.argv) > 1 else "/home/z00603376/real_kv_dump.pt"
LAMBDAS = [-3.0, -1.0]


def main():
    d = torch.load(DUMP, map_location="cpu")
    T = d["token_ids"].numel()
    print(f"prompt T={T}")
    print(f"{'layer':>6}{'qmode':>9}{'lam':>5}  {'kernel':>18}")
    for L in sorted(d["layers"]):
        q_full = d["layers"][L]["q"].to(DEVICE)
        k = d["layers"][L]["k"].to(DEVICE)
        v = d["layers"][L]["v"].to(DEVICE)
        for qmode in ("decode", "prefill"):
            if qmode == "decode":
                q = q_full[-1:]          # last token attends all kv
                q_lens, kv_lens = [1], [T]
            else:
                q = q_full
                q_lens, kv_lens = [T], [T]
            for lam in LAMBDAS:
                try:
                    _, _, stats = run_custom(q, k, v, q_lens, kv_lens,
                                             24, 4, SCALE, sparse_lambda=lam,
                                             causal=(qmode == "prefill"),
                                             sparse_stats_flag=True)
                    ks, kt = int(stats[0]), int(stats[1])
                    print(f"{L:>6}{qmode:>9}{lam:>5}  {ks:>7}/{kt:<7} {ks / max(kt, 1):>6.1%}",
                          flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"{L:>6}{qmode:>9}{lam:>5}  ERROR: {str(e)[:70]}", flush=True)


if __name__ == "__main__":
    main()
