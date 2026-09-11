#!/usr/bin/env python3
"""Dump REAL K/V (and Q) for full-attention layers from a real prompt.

Offline single-layer recompute (no server): embedding -> per-layer W_qkv ->
QK-norm -> RoPE(theta=1e7) -> cast bf16, matching the serving projection of
the custom FIA op. Approximation note: the same pre-computed token embedding
is fed to every probed layer, so mid/deep layers (31/63) capture weight+logit
geometry but not the true residual-stream distribution. Layer 3 (first full-
attention layer) is the most faithful.
"""
import json
import math
import sys

sys.path.insert(0, "/home/z00603376/fia/vllm-ascend/tests/fused_infer_attention_score")
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODEL = "/mnt/weight/Qwen3.5-27B"
CFG = json.load(open(f"{MODEL}/config.json"))["text_config"]
IDX = json.load(open(f"{MODEL}/model.safetensors.index.json"))
N_H, N_KV, D, HID = (CFG["num_attention_heads"], CFG["num_key_value_heads"],
                     CFG["head_dim"], CFG["hidden_size"])
EPS = CFG["rms_norm_eps"]
THETA = CFG["rope_parameters"]["rope_theta"]
PARTIAL = CFG["rope_parameters"]["partial_rotary_factor"]
ROT_DIM = int(D * PARTIAL)
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/z00603376/real_kv_dump.pt"
LAYERS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["3", "31", "63"])]
PROMPT_LEN = int(sys.argv[3]) if len(sys.argv) > 3 else 4096


def w(name):
    f = IDX["weight_map"][name]
    with safe_open(f"{MODEL}/{f}", "pt") as sf:
        return sf.get_tensor(name).float()


def rms(x, g):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g


def rope(x, rot):
    # x: [T, H, D]; rotate first rot dims (non-interleaved), pass rest
    T = x.size(0)
    pos = torch.arange(T).float()
    inv = 1.0 / (THETA ** (torch.arange(0, rot, 2).float() / rot))
    freqs = torch.outer(pos, inv)               # [T, rot/2]
    cos, sin = freqs.cos()[:, None, :], freqs.sin()[:, None, :]
    x1, x2 = x[..., : rot // 2], x[..., rot // 2: rot]
    x_pass = x[..., rot:]
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos
    return torch.cat([r1, r2, x_pass], dim=-1)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    filler = ("The logistics archive contains numbered records of warehouse "
              "inventory across regional distribution centers for audit purposes. ")
    text = filler * (PROMPT_LEN // 12)
    text += " The magic access code for the north gate is ZEBRA-7741. " + filler * 40
    text += "\n\nQuestion: What is the magic access code for the north gate?"
    ids = tok(text, return_tensors="pt").input_ids[0]
    ids = ids[:PROMPT_LEN]
    T = ids.size(0)
    print(f"prompt tokens: {T}, layers: {LAYERS}", flush=True)

    emb = w("model.language_model.embed_tokens.weight")[ids]  # [T, HID]

    dump = {"token_ids": ids, "layers": {}}
    for L in LAYERS:
        p = f"model.language_model.layers.{L}.self_attn."
        Wq, Wk, Wv = w(p + "q_proj.weight"), w(p + "k_proj.weight"), w(p + "v_proj.weight")
        qg, kg = w(p + "q_norm.weight"), w(p + "k_norm.weight")
        q = emb @ Wq.T                                # [T, N_H*2*D] (with gate)
        q = q.view(T, N_H, 2 * D)[..., :D]            # drop output gate half
        k = (emb @ Wk.T).view(T, N_KV, D)
        v = (emb @ Wv.T).view(T, N_KV, D)
        q = rms(q, qg)
        k = rms(k, kg)
        q = rope(q, ROT_DIM)
        k = rope(k, ROT_DIM)
        dump["layers"][L] = {"q": q.bfloat16(), "k": k.bfloat16(), "v": v.bfloat16()}
        print(f"layer {L}: q{tuple(q.shape)} k{tuple(k.shape)} dumped", flush=True)
        del Wq, Wk, Wv
    torch.save(dump, OUT)
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
