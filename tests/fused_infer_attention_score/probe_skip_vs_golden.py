#!/usr/bin/env python3
"""Single-op reproduction: why is the skip ratio ~0 on synthetic data?

Runs the SAME tensors through the NPU custom op (sparse_stats return) and the
CPU golden (kernel-exact skip simulation) and compares their skip ratios.

Criterion (golden _detect_sparse_block): a KV block is skipped when
    block_half_max - running_global_max < sparse_lamda
i.e. an ABSOLUTE logit gap of |lambda| nats. The scaled score std for iid
data is ~1.0 (scores = q.k * 1/sqrt(D)), so half-block maxes sit only ~2
nats below the global max and never clear |lambda|=3. Modes here bracket the
explanation:

  iid       k ~ N(0,1): scaled logit std ~1  -> tiny gaps -> no skip
  clustered topic vectors: near-zero score contrast           -> no skip
  iid_hi    k ~ N(0,8): scaled logit std ~8 (real-model range)-> skips
  bimodal   strong mid segment + near-zero tail (case-6 style) -> skips
"""
import math
import sys

sys.path.insert(0, "/home/z00603376/fia/vllm-ascend/tests/fused_infer_attention_score")
import torch  # noqa: E402
from validate_post_cleanup import DEVICE, run_custom, _cumsum  # noqa: E402
from blasst_golden_tnd import BlasstGoldenTND  # noqa: E402

D = 256
H = 12
SCALE = 1.0 / math.sqrt(D)
DT = torch.float16
KV = 4096
LAMBDAS = [-3.0, -1.0]


def gen(mode, q_len, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.randn(KV, H, D, generator=g)
    v = torch.randn(KV, H, D, generator=g)
    q = torch.randn(q_len, H, D, generator=g)
    if mode == "iid_hi":
        k = k * 8.0
    elif mode == "bimodal":
        k[KV // 2:KV // 2 + 512] *= 8.0   # strong mid segment
        k[int(KV * 0.75):] *= 0.01        # near-dead tail
    elif mode == "clustered":
        topics = torch.randn(4, 1, D, generator=g)
        topics = topics / topics.norm(dim=-1, keepdim=True)
        nblk = (KV + 127) // 128
        assign = torch.arange(nblk) % 4
        base = topics[assign].repeat_interleave(128, dim=0)[:KV]
        k = (base + 0.05 * torch.randn(KV, 1, D, generator=g)).repeat(1, H, 1)
        # query sits on topic 0 for every head -> off-topic blocks skippable
        q = topics[0].expand(q_len, H, D) \
            + 0.05 * torch.randn(q_len, H, D, generator=g)
    return q.to(DT).to(DEVICE), k.to(DT).to(DEVICE), v.to(DT).to(DEVICE)


def main():
    import time
    print(f"{'mode':<10}{'q':>7}{'lam':>5}  {'kernel':>16}  {'golden':>16}  verdict")
    cells = [(m, qm) for m in ("iid", "clustered", "iid_hi", "bimodal")
             for qm in ("decode", "prefill")]
    for mode, qmode in cells:
        if qmode == "prefill" and mode in ("clustered",):
            continue  # decode row already establishes agreement for this mode
        q_len = 1 if qmode == "decode" else KV
        q, k, v = gen(mode, q_len, seed=42 + q_len)
        for lam in LAMBDAS:
            t0 = time.time()
            _, _, stats = run_custom(q, k, v, [q_len], [KV], H, H, SCALE,
                                     sparse_lambda=lam,
                                     sparse_stats_flag=True)
            torch.npu.synchronize()
            ks, kt = int(stats[0]), int(stats[1])
            t1 = time.time()
            golden = BlasstGoldenTND(num_heads=H, num_key_value_heads=H,
                                     head_dim=D, scale=SCALE,
                                     block_size=32, sparse_lamda=lam,
                                     rowloop_rows=16)
            _, _, info = golden.forward_blasst_kernel(
                q.float().cpu(), k.float().cpu(), v.float().cpu(),
                torch.tensor(_cumsum([q_len])),
                torch.tensor(_cumsum([KV])))
            gs, gt = info["skipped_blocks"], info["total_blocks"]
            t2 = time.time()
            kr = ks / max(kt, 1)
            gr = gs / max(gt, 1)
            agree = "MATCH" if abs(kr - gr) < 0.02 else "DIFF!"
            print(f"{mode:<10}{qmode:>7}{lam:>5}  {ks:>6}/{kt:<6}{kr:>5.1%}"
                  f"  {gs:>6}/{gt:<6}{gr:>5.1%}  {agree}"
                  f"  (npu {t1-t0:.0f}s / golden {t2-t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
