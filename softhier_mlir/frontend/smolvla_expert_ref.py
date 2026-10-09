"""Host reference for the SmolVLA action expert: run lerobot's SmolVLAPolicy code (eager attention) on a fixed
synthetic observation and dump everything the device program consumes or is compared against.

    /app/models/lerobot_venv/bin/python -m softhier_mlir.frontend.smolvla_expert_ref --ckpt /app/models/smolvla_base \
        --out /app/models/smolvla_base/expert_ref.npz [--vlm-dtype fp32|bf16] [--phase kv|expert|all]

Needs lerobot (its modeling_smolvla / smolvlm_with_expert are the reference; a venv with
`pip install lerobot "transformers<5" num2words`), torch, and hub access for the SmolVLM2-500M config + tokenizer
(the VLM weights come from the checkpoint, not the hub: `load_vlm_weights=False`).

Memory: the shared host has 7 GB and is full of simulators, so the policy is built on the meta device and only the
tensors a phase needs are materialised: phase `kv` runs the frozen VLM (vision tower + connector first, freed, then
the text model) and writes <out>.kv.npz; phase `expert` loads only the expert + the action projections (fp32), reads
that file, runs the 10 denoising steps with lerobot's `denoise_step` and writes <out>. `all` = both in one process.

What is dumped (all fp32 unless noted), layer index l = 0..15 of the 16 VLM/expert layers:
  images            [3, 3, 512, 512] fp16   the three camera inputs in SigLIP's [-1, 1] range (seeded uniform noise)
  lang_tokens/mask  [48]                     tokenizer(prompt, padding max_length 48, right)
  state             [32]                     6 real dims ~ N(0,1), zero padded (MEAN_STD-normalised convention)
  noise             [50, 32]                 x_0 of the flow (torch.randn, seed)
  prefix_valid      [241]                    prefix_pad_masks: 1 for real tokens, 0 for language padding
  kv_k_<l>, kv_v_<l> [241, 320]              the VLM prefix KV cache of layer l exactly as VLMWithExpertModel stores it:
                                             key_states AFTER RoPE (positions cumsum(pad)-1), value_states, both
                                             [tokens, kv_head * 64 + d] (5 kv heads, head-major columns)
  time_emb          [10, 720]                create_sinusoidal_pos_embedding(t_s) for t_s = 1 - s/10
  xt                [11, 50, 32]             x_t before step s (xt[0] = noise, xt[10] = the action chunk)
  vt                [10, 50, 32]             v_t of step s
  actions           [50, 32]                 == xt[10]
  s0_emb            [50, 720]                step-0 suffix embeddings (action_time_mlp_out)
  s0_h_<l>          [50, 720]                step-0 residual stream after expert layer l
  s0_att_<l>        [50, 960]                step-0 attention output (o_proj input) of layer l
  s0_q_<l>          [50, 960]                step-0 q_proj output (before RoPE) of layer l
  s0_fin            [50, 720]                step-0 final-norm output
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
from pathlib import Path

import numpy as np
import torch

PROMPT = "Pick up the red cube and put it in the box"


def build_policy(ckpt: str):
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    raw = json.load(open(Path(ckpt) / "config.json"))
    raw.pop("type", None)
    for key in ("input_features", "output_features"):
        raw[key] = {k: PolicyFeature(type=FeatureType[v["type"]], shape=tuple(v["shape"])) for k, v in raw[key].items()}
    raw["normalization_mapping"] = {k: NormalizationMode[v] for k, v in raw["normalization_mapping"].items()}
    names = {f.name for f in dataclasses.fields(SmolVLAConfig)}
    cfg = SmolVLAConfig(**{k: v for k, v in raw.items() if k in names})
    cfg.load_vlm_weights = False          # VLM weights come from the checkpoint (no 1 GB hub download)
    cfg.device = "cpu"
    with torch.device("meta"):
        policy = SmolVLAPolicy(cfg)
    policy.model.vlm_with_expert.vlm.lm_head = torch.nn.Identity()
    return cfg, policy.eval()


def materialise(policy, ckpt: str, want, dtype_of) -> None:
    """Load the checkpoint tensors whose key satisfies want(key) into the meta-built policy (assign), rebuild the
    non-persistent buffers of the loaded part."""
    from safetensors import safe_open
    tensors = {}
    with safe_open(str(Path(ckpt) / "model.safetensors"), "pt") as f:
        for k in f.keys():
            if k.endswith("lm_head.weight") or not want(k):
                continue
            tensors[k] = f.get_tensor(k).to(dtype_of(k))
    missing, unexpected = policy.load_state_dict(tensors, strict=False, assign=True)
    assert not unexpected, unexpected[:5]
    assert not [k for k in missing if want(k) and "lm_head" not in k], [k for k in missing if want(k)][:5]
    del tensors
    for name, mod in policy.named_modules():
        for bname, buf in list(mod.named_buffers(recurse=False)):
            if not buf.is_meta:
                continue
            if bname == "position_ids":           # SigLIP vision embeddings: arange(num_patches)
                mod.register_buffer(bname, torch.arange(buf.shape[-1]).expand(buf.shape).clone(), persistent=False)
            elif bname in ("inv_freq", "original_inv_freq"):   # HF rotary (unused: lerobot applies its own RoPE)
                mod.register_buffer(bname, torch.zeros(buf.shape, dtype=torch.float32), persistent=False)
            else:
                raise RuntimeError(f"meta buffer left: {name}.{bname} {tuple(buf.shape)}")


def inputs(cfg, tok, seed: int, prompt: str, cams: int = 3, size: int = 512):
    """cams camera images of size x size (SigLIP's [-1, 1] range, fp16-representable), the tokenized prompt, state, noise.
    size 512 = lerobot's resize_imgs_with_padding (1024 SigLIP tokens -> 64 image tokens per camera); 256 = 256 SigLIP
    tokens -> 16 image tokens (SmolVLM's bucketed position ids: patch (i, j) of the 16 x 16 grid takes the position row
    of the 32 x 32 table that `vision_position_ids` gives)."""
    g = torch.Generator().manual_seed(seed)
    images = [(torch.rand(1, 3, size, size, generator=g) * 2 - 1).half().float() for _ in range(cams)]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(cams)]
    enc = tok(prompt, padding="max_length", padding_side="right", max_length=cfg.tokenizer_max_length, truncation=True, return_tensors="pt")
    lang_tokens, lang_masks = enc["input_ids"], enc["attention_mask"].bool()
    state = torch.zeros(1, 32); state[0, :6] = torch.randn(6, generator=g)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=g)
    return images, img_masks, lang_tokens, lang_masks, state, noise


def vision_position_ids(size: int, patch: int = 16, image_size: int = 512) -> np.ndarray:
    """The position-table rows SmolVLM's vision embeddings use for a full (unpadded) size x size image: the code of
    SmolVLMVisionEmbeddings.forward (bucketize of the fractional patch coordinates into the 32 x 32 grid)."""
    side, n = image_size // patch, size // patch
    boundaries = torch.arange(1 / side, 1.0, 1 / side)
    idx = torch.arange(n, dtype=torch.float32)
    frac = idx / n * (1 - 1e-6)
    b = torch.bucketize(frac, boundaries, right=True)
    return (b[:, None] * side + b[None, :]).flatten().numpy()


def phase_kv(a) -> dict:
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    cfg, policy = build_policy(a.ckpt)
    m = policy.model; we = m.vlm_with_expert
    vlm_dtype = torch.float32 if a.vlm_dtype == "fp32" else torch.bfloat16
    images, img_masks, lang_tokens, lang_masks, state, noise = inputs(cfg, we.processor.tokenizer, a.seed, a.prompt,
                                                                      getattr(a, "cams", 3), getattr(a, "img_size", 512))
    out = {"images": torch.cat(images).half().numpy(), "vis_pos_ids": vision_position_ids(images[0].shape[-1]), "lang_tokens": lang_tokens[0].numpy(), "lang_mask": lang_masks[0].numpy(),
           "state": state[0].numpy(), "noise": noise[0].numpy(), "prompt": np.array(a.prompt), "vlm_dtype": np.array(a.vlm_dtype)}
    # 1. vision tower + connector -> image embeddings (then freed: the host has no room for both halves of the VLM)
    materialise(policy, a.ckpt, lambda k: ".vlm.model.vision_model." in k or ".vlm.model.connector." in k, lambda k: vlm_dtype)
    with torch.no_grad():
        img_embs = []
        for c, img in enumerate(images):      # == we.embed_image(img) (patch_attention_mask None), keeping the SigLIP output
            vis = we.get_vlm_model().vision_model(pixel_values=img.to(vlm_dtype), patch_attention_mask=None).last_hidden_state
            img_embs.append(we.get_vlm_model().connector(vis).clone())
            out[f"vis_out_{c}"] = vis[0].float().numpy().copy()          # post-LN SigLIP output [tokens, 768]
            out[f"img_emb_{c}"] = img_embs[-1][0].float().numpy().copy()   # connector output (before the sqrt(960) scaling)
        assert torch.equal(img_embs[0], we.embed_image(images[0])), "manual vision + connector path != embed_image"
    print(f"[ref] image embeddings {tuple(img_embs[0].shape)} (vision+connector {vlm_dtype})", flush=True)
    vm = we.get_vlm_model()
    vm.vision_model = torch.nn.Identity(); vm.connector = torch.nn.Identity(); gc.collect()
    cache = list(img_embs)
    we.embed_image = lambda img: cache.pop(0)      # embed_prefix runs unchanged on the cached embeddings
    # 2. text model + state_proj -> prefix KV cache
    materialise(policy, a.ckpt, lambda k: ".vlm.model.text_model." in k or "model.state_proj." in k,
                lambda k: vlm_dtype if ".vlm." in k else torch.float32)
    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, pkv = we.forward(attention_mask=prefix_att_2d, position_ids=prefix_pos, past_key_values=None,
                            inputs_embeds=[prefix_embs, None], use_cache=cfg.use_cache, fill_kv_cache=True)
    Lp = prefix_pad_masks.shape[1]
    out["prefix_embs"] = prefix_embs[0].float().numpy()
    out["prefix_valid"] = prefix_pad_masks[0].float().numpy()
    out["n_valid"] = np.array(int(prefix_pad_masks.sum()))
    for l in range(we.num_vlm_layers):
        out[f"kv_k_{l}"] = pkv[l]["key_states"][0].reshape(Lp, -1).float().numpy()
        out[f"kv_v_{l}"] = pkv[l]["value_states"][0].reshape(Lp, -1).float().numpy()
    print(f"[ref] prefix length {Lp}, valid tokens {int(prefix_pad_masks.sum())}, KV per layer {out['kv_k_0'].shape}, "
          f"|K| max {max(np.abs(out[f'kv_k_{l}']).max() for l in range(16)):.2f} |V| max {max(np.abs(out[f'kv_v_{l}']).max() for l in range(16)):.2f}")
    np.savez(a.out + ".kv.npz", **out)
    print(f"[ref] wrote {a.out}.kv.npz", flush=True)
    del policy, pkv, prefix_embs; gc.collect()
    return out


def phase_expert(a, kv: dict | None) -> None:
    from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding
    if kv is None:
        kv = dict(np.load(a.out + ".kv.npz"))
    cfg, policy = build_policy(a.ckpt)
    m = policy.model; we = m.vlm_with_expert
    materialise(policy, a.ckpt, lambda k: ".lm_expert." in k or k.startswith("model.action_"), lambda k: torch.float32)
    print(f"[ref] expert layers {we.num_expert_layers}, hidden {we.expert_hidden_size}, heads {we.num_attention_heads}/{we.num_key_value_heads}, "
          f"head_dim {we.vlm.config.text_config.head_dim}, rms eps {we.lm_expert.config.rms_norm_eps}, intermediate {we.lm_expert.config.intermediate_size}", flush=True)
    Lp = int(kv["prefix_valid"].shape[0])
    pkv = {l: {"key_states": torch.from_numpy(kv[f"kv_k_{l}"]).reshape(1, Lp, we.num_key_value_heads, -1),
               "value_states": torch.from_numpy(kv[f"kv_v_{l}"]).reshape(1, Lp, we.num_key_value_heads, -1)} for l in range(we.num_vlm_layers)}
    prefix_pad_masks = torch.from_numpy(kv["prefix_valid"]).bool()[None]
    noise = torch.from_numpy(kv["noise"])[None]
    s0: dict[str, np.ndarray] = {}
    rec = {"on": False}

    def keep(name, inp=False):
        def h(mod, i, o):
            if rec["on"]:
                t = i[0] if inp else (o if not isinstance(o, tuple) else o[0])
                s0[name] = t.detach()[0].float().numpy().copy()
        return h
    for l, layer in enumerate(we.lm_expert.layers):
        layer.post_attention_layernorm.register_forward_hook(keep(f"s0_hatt_{l}", True))
        layer.mlp.register_forward_hook(keep(f"s0_mlp_{l}"))
        layer.self_attn.o_proj.register_forward_hook(keep(f"s0_att_{l}", True))
        layer.self_attn.q_proj.register_forward_hook(keep(f"s0_q_{l}"))
    we.lm_expert.norm.register_forward_hook(keep("s0_fin"))
    m.action_time_mlp_out.register_forward_hook(keep("s0_emb"))
    num_steps, dt = cfg.num_steps, -1.0 / cfg.num_steps
    xt, vt, temb = [kv["noise"].copy()], [], []
    x_t = noise
    with torch.no_grad():
        for step in range(num_steps):
            t = 1.0 + step * dt
            tt = torch.tensor(t, dtype=torch.float32).expand(1)
            temb.append(create_sinusoidal_pos_embedding(tt, we.expert_hidden_size, cfg.min_period, cfg.max_period, device=tt.device)[0].float().numpy())
            rec["on"] = step == 0
            v_t = m.denoise_step(x_t=x_t, prefix_pad_masks=prefix_pad_masks, past_key_values=pkv, timestep=tt)
            rec["on"] = False
            x_t = x_t + dt * v_t
            vt.append(v_t[0].numpy().copy()); xt.append(x_t[0].numpy().copy())
            print(f"[ref] step {step} t={t:.1f} |v_t| max {np.abs(vt[-1]).max():.3f} |x_t| max {np.abs(xt[-1]).max():.3f}", flush=True)
    for l in range(we.num_expert_layers):
        s0[f"s0_h_{l}"] = s0[f"s0_hatt_{l}"] + s0[f"s0_mlp_{l}"]
    out = dict(kv)
    out.update({"time_emb": np.stack(temb), "xt": np.stack(xt), "vt": np.stack(vt), "actions": xt[-1]})
    out.update(s0)
    out["meta"] = np.array([cfg.chunk_size, cfg.max_action_dim, num_steps, Lp, int(kv["n_valid"])], dtype=np.int64)
    np.savez(a.out, **out)
    print(f"[ref] actions: mean {xt[-1].mean():.4f} std {xt[-1].std():.4f} max |a| {np.abs(xt[-1]).max():.3f}")
    print(f"[ref] wrote {a.out} ({len(out)} arrays)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="/app/models/smolvla_base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--vlm-dtype", default="fp32", choices=["fp32", "bf16"], help="dtype of the frozen VLM (prefix KV producer); the expert is always fp32")
    ap.add_argument("--phase", default="all", choices=["kv", "expert", "all"])
    ap.add_argument("--cams", type=int, default=3, help="cameras (the end-to-end chain also runs 1)")
    ap.add_argument("--img-size", type=int, default=512, help="camera image side: 512 (1024 SigLIP tokens) or 256 (256 tokens)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    kv = phase_kv(a) if a.phase in ("kv", "all") else None
    if a.phase in ("expert", "all"):
        phase_expert(a, kv)


if __name__ == "__main__":
    main()
