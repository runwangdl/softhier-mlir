"""On-chip RSSM imagination (runtime/sh_wm.inc.c, docs/WORLD_MODEL.md): host side.

The model is the DreamerV3-style RSSM + actor of world-model-on-edge (`wm/rssm.py`, bit-exact on GAP9 through
Deeploy): MLP encoder, LayerNorm-GRU (update bias - 1), 16 x 16 categorical latent with unimix 0.01, ReLU MLPs,
actor head tanh(mean). Two configurations: full (deter 128, stoch 16 x 16, hidden / units 128) and nano (deter 64,
stoch 8 x 8, hidden / units 64); trained weights in /app/world-model-on-edge/models/rssm_{step,nano}_trained (obs 51,
act 3).

  prepare (system python: torch)
      python3 -m softhier_mlir.frontend.wm_rssm prepare --model step --K 128 --H 10 --out /app/models/wm/rssm_step.npz
      loads the trained state dict (random weights with seed 0 if absent), draws K initial states / actions / H
      observations, runs the torch module (rssm.py imported from world-model-on-edge): one posterior step
      (RSSM.step) and an H-step open-loop imagination rollout (RSSM.prior + actor), and stores the weights (sd_*),
      inputs and torch references (ref_*).
  pack / np_* (numpy only): the weight blob in the kernel's layout and numpy twins of the kernel's algebra
      (fp32, and the fp16-floor variant that rounds every stored tensor like the device).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

WM_REPO = Path("/app/world-model-on-edge")
LN_EPS = 1e-3
UNIMIX = 0.01
TABLE = ["w_in", "g_in", "b_in", "w_g", "g_g", "b_g", "w_io", "g_io", "b_io", "w_is", "b_is", "w_a1", "g_a1", "b_a1",
         "w_a2", "g_a2", "b_a2", "w_ao", "b_ao", "w_e1", "g_e1", "b_e1", "w_e2", "g_e2", "b_e2", "w_oo", "g_oo", "b_oo",
         "w_os", "b_os"]
POSTERIOR_ONLY = {"w_e1", "g_e1", "b_e1", "w_e2", "g_e2", "b_e2", "w_oo", "g_oo", "b_oo", "w_os", "b_os"}


@dataclass
class Cfg:
    obs: int = 51
    act: int = 3
    deter: int = 128
    stoch: int = 16
    classes: int = 16
    hidden: int = 128
    units: int = 128

    @property
    def S(self):
        return self.stoch * self.classes

    @property
    def AP(self):
        return (self.act + 3) & ~3

    @property
    def OP(self):
        return (self.obs + 3) & ~3

    @property
    def AO(self):
        return (2 * self.act + 3) & ~3


CFGS = {"step": Cfg(), "nano": Cfg(deter=64, stoch=8, classes=8, hidden=64, units=64)}


def macs_per_step(c: Cfg, posterior: bool = False) -> int:
    """MACs per trajectory per step of the kernel (padded shapes as run on RedMulE)."""
    m = (c.S + c.AP) * c.hidden + (c.hidden + c.deter) * 3 * c.deter
    m += (c.deter + c.units) * c.hidden + c.hidden * c.S if posterior else c.deter * c.hidden + c.hidden * c.S
    if posterior:
        m += c.OP * c.units + c.units * c.units
    m += (c.S + c.deter) * c.units + c.units * c.units + c.units * c.AO
    return m


# ----------------------------------------------------------------------------- weight blob
def lib_weights(c: Cfg, sd: dict) -> dict[str, np.ndarray]:
    """state dict (torch names, fp32) -> {table name: fp32 array in the kernel layout ([in, out], padded rows/cols)}."""
    T = lambda a: np.ascontiguousarray(np.asarray(a, np.float32).T)  # noqa: E731
    S, D = c.S, c.deter
    w = {}
    win = np.zeros((S + c.AP, c.hidden), np.float32); win[:S + c.act] = T(sd["img_in.lin.weight"])
    w["w_in"], w["g_in"], w["b_in"] = win, sd["img_in.ln.weight"], sd["img_in.ln.bias"]
    w["w_g"], w["g_g"] = T(sd["cell.layer.weight"]), sd["cell.ln.weight"]
    bg = np.asarray(sd["cell.ln.bias"], np.float32).copy(); bg[2 * D:] -= 1.0          # sigmoid(update - 1)
    w["b_g"] = bg
    w["w_io"], w["g_io"], w["b_io"] = T(sd["img_out.lin.weight"]), sd["img_out.ln.weight"], sd["img_out.ln.bias"]
    w["w_is"], w["b_is"] = T(sd["img_stat.weight"]), sd["img_stat.bias"]
    w["w_a1"], w["g_a1"], w["b_a1"] = T(sd["actor.0.lin.weight"]), sd["actor.0.ln.weight"], sd["actor.0.ln.bias"]
    w["w_a2"], w["g_a2"], w["b_a2"] = T(sd["actor.1.lin.weight"]), sd["actor.1.ln.weight"], sd["actor.1.ln.bias"]
    wao = np.zeros((c.units, c.AO), np.float32); wao[:, :2 * c.act] = T(sd["actor_out.weight"])
    bao = np.zeros(c.AO, np.float32); bao[:2 * c.act] = sd["actor_out.bias"]
    w["w_ao"], w["b_ao"] = wao, bao
    we1 = np.zeros((c.OP, c.units), np.float32); we1[:c.obs] = T(sd["enc.0.lin.weight"])
    w["w_e1"], w["g_e1"], w["b_e1"] = we1, sd["enc.0.ln.weight"], sd["enc.0.ln.bias"]
    w["w_e2"], w["g_e2"], w["b_e2"] = T(sd["enc.1.lin.weight"]), sd["enc.1.ln.weight"], sd["enc.1.ln.bias"]
    w["w_oo"], w["g_oo"], w["b_oo"] = T(sd["obs_out.lin.weight"]), sd["obs_out.ln.weight"], sd["obs_out.ln.bias"]
    w["w_os"], w["b_os"] = T(sd["obs_stat.weight"]), sd["obs_stat.bias"]
    return {k: np.asarray(v, np.float32) for k, v in w.items()}


def pack(c: Cfg, sd: dict, posterior: bool = True) -> tuple[np.ndarray, dict[str, int]]:
    """The kernel's weight blob: uint32 table[32] (byte offsets of TABLE entries, 0 = absent; table[31] = blob bytes),
    then each entry as fp16, 64-byte aligned. Returns (uint16 array of the blob, offsets)."""
    w = lib_weights(c, sd)
    off, cur, parts = {}, 128, []
    for name in TABLE:
        if not posterior and name in POSTERIOR_ONLY:
            off[name] = 0
            continue
        a = w[name].astype(np.float16).reshape(-1)
        off[name] = cur
        pad = (-(a.nbytes)) % 64
        parts.append((cur, a))
        cur += a.nbytes + pad
    blob = np.zeros(cur // 2, np.uint16)
    tab = np.zeros(32, np.uint32)
    for i, name in enumerate(TABLE):
        tab[i] = off[name]
    tab[31] = cur
    blob[:64] = tab.view(np.uint16)
    for o, a in parts:
        blob[o // 2:o // 2 + a.size] = a.view(np.uint16)
    return blob, off



def entry_shapes(c: Cfg) -> dict[str, tuple[int, ...]]:
    """shape of every blob entry (TABLE order)"""
    S, D, Hd, U = c.S, c.deter, c.hidden, c.units
    sh = {"w_in": (S + c.AP, Hd), "g_in": (Hd,), "b_in": (Hd,), "w_g": (Hd + D, 3 * D), "g_g": (3 * D,), "b_g": (3 * D,),
          "w_io": (D, Hd), "g_io": (Hd,), "b_io": (Hd,), "w_is": (Hd, S), "b_is": (S,), "w_a1": (S + D, U), "g_a1": (U,),
          "b_a1": (U,), "w_a2": (U, U), "g_a2": (U,), "b_a2": (U,), "w_ao": (U, c.AO), "b_ao": (c.AO,),
          "w_e1": (c.OP, U), "g_e1": (U,), "b_e1": (U,), "w_e2": (U, U), "g_e2": (U,), "b_e2": (U,),
          "w_oo": (D + U, Hd), "g_oo": (Hd,), "b_oo": (Hd,), "w_os": (Hd, S), "b_os": (S,)}
    return sh


def pack_bytes(c: Cfg, posterior: bool = True) -> int:
    """size of pack()'s blob without building it"""
    cur = 128
    for name, shp in entry_shapes(c).items():
        if not posterior and name in POSTERIOR_ONLY:
            continue
        cur += (int(np.prod(shp)) * 2 + 63) & ~63
    return cur

# ----------------------------------------------------------------------------- numpy twins
def _r16(a):
    return np.asarray(a, np.float32).astype(np.float16).astype(np.float32)


class Twin:
    """The kernel's algebra in numpy. fp16=True rounds every tensor the device stores (GEMM outputs, LN outputs,
    gates' h', z', a') to fp16 and uses fp16 weights; fp16=False is the fp32 model (== torch to ~1e-6)."""

    def __init__(self, c: Cfg, sd: dict, fp16: bool = False):
        self.c, self.fp16 = c, fp16
        w = lib_weights(c, sd)
        self.w = {k: (_r16(v) if fp16 else v) for k, v in w.items()}
        self.r = _r16 if fp16 else (lambda a: np.asarray(a, np.float32))

    def mm(self, x, k):
        return self.r(x @ self.w[k])

    def ln(self, x, g, b, relu=True):
        m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
        y = (x - m) / np.sqrt(v + LN_EPS) * self.w[g] + self.w[b]
        return self.r(np.maximum(y, 0) if relu else y)

    def latent(self, logits):
        c = self.c
        l = logits.reshape(-1, c.stoch, c.classes)
        e = np.exp(l - l.max(-1, keepdims=True)); p = e / e.sum(-1, keepdims=True)
        return self.r(((1 - UNIMIX) * p + UNIMIX / c.classes).reshape(-1, c.S))

    def gru(self, x, h):
        D = self.c.deter
        p = self.ln(self.mm(np.concatenate([x, h], -1), "w_g"), "g_g", "b_g", relu=False)
        r = 1 / (1 + np.exp(-p[:, :D])); cand = np.tanh(r * p[:, D:2 * D]); u = 1 / (1 + np.exp(-p[:, 2 * D:]))
        return self.r(h + u * (cand - h))

    def actor(self, z, h):
        c = self.c
        a = self.ln(self.mm(np.concatenate([z, h], -1), "w_a1"), "g_a1", "b_a1")
        a = self.ln(self.mm(a, "w_a2"), "g_a2", "b_a2")
        o = self.mm(a, "w_ao")[:, :c.act] + self.w["b_ao"][:c.act]
        return self.r(np.tanh(o))

    def _za(self, z, a):
        c = self.c
        za = np.zeros((z.shape[0], c.S + c.AP), np.float32); za[:, :c.S] = z; za[:, c.S:c.S + c.act] = a
        return za

    def prior(self, h, z, a):
        x = self.ln(self.mm(self._za(z, a), "w_in"), "g_in", "b_in")
        h = self.gru(x, h)
        y = self.ln(self.mm(h, "w_io"), "g_io", "b_io")
        return h, self.latent(self.mm(y, "w_is") + self.w["b_is"])

    def step(self, obs, h, z, a):
        c = self.c
        ob = np.zeros((obs.shape[0], c.OP), np.float32); ob[:, :c.obs] = obs
        e = self.ln(self.mm(self.ln(self.mm(ob, "w_e1"), "g_e1", "b_e1"), "w_e2"), "g_e2", "b_e2")
        x = self.ln(self.mm(self._za(z, a), "w_in"), "g_in", "b_in")
        h = self.gru(x, h)
        y = self.ln(self.mm(np.concatenate([h, e], -1), "w_oo"), "g_oo", "b_oo")
        z = self.latent(self.mm(y, "w_os") + self.w["b_os"])
        return h, z, self.actor(z, h)

    def rollout(self, h, z, a, H, obs=None):
        """H steps; obs [H, K, obs] -> posterior steps, else imagination. Returns hs, zs, as ([H, K, .])."""
        hs, zs, as_ = [], [], []
        h, z, a = (self.r(v) for v in (h, z, a))
        for t in range(H):
            if obs is None:
                h, z = self.prior(h, z, a)
                a = self.actor(z, h)
            else:
                h, z, a = self.step(obs[t], h, z, a)
            hs.append(h); zs.append(z); as_.append(a)
        return np.stack(hs), np.stack(zs), np.stack(as_)


def random_sd(c: Cfg, seed: int = 0) -> dict:
    """rssm.py's parameter shapes with PyTorch-like default init (uniform +-1/sqrt(fan_in), LN 1 / 0)."""
    rng = np.random.default_rng(seed)
    S = c.S
    lin = lambda o, i: rng.uniform(-1, 1, (o, i)).astype(np.float32) / np.sqrt(i)  # noqa: E731
    sd = {}
    for pre, o, i in (("enc.0", c.units, c.obs), ("enc.1", c.units, c.units), ("img_in", c.hidden, S + c.act),
                      ("obs_out", c.hidden, c.deter + c.units), ("img_out", c.hidden, c.deter), ("actor.0", c.units, S + c.deter),
                      ("actor.1", c.units, c.units)):
        sd[f"{pre}.lin.weight"] = lin(o, i); sd[f"{pre}.ln.weight"] = np.ones(o, np.float32); sd[f"{pre}.ln.bias"] = np.zeros(o, np.float32)
    sd["cell.layer.weight"] = lin(3 * c.deter, c.hidden + c.deter)
    sd["cell.ln.weight"], sd["cell.ln.bias"] = np.ones(3 * c.deter, np.float32), np.zeros(3 * c.deter, np.float32)
    for pre, i in (("obs_stat", c.hidden), ("img_stat", c.hidden)):
        sd[f"{pre}.weight"] = lin(S, i); sd[f"{pre}.bias"] = rng.uniform(-1, 1, S).astype(np.float32) / np.sqrt(i)
    sd["actor_out.weight"] = lin(2 * c.act, c.units); sd["actor_out.bias"] = np.zeros(2 * c.act, np.float32)
    return sd


def initial_states(c: Cfg, K: int, H: int, seed: int = 0):
    """K initial (h, z, a) and H observations: h = tanh(N(0,1)), z = unimix of a random distribution per group,
    a uniform in [-1, 1], obs N(0, 1) (the inputs of export_imagine.py, plus observations)."""
    rng = np.random.default_rng(seed)
    h = np.tanh(rng.normal(size=(K, c.deter))).astype(np.float32)
    z0 = rng.random((K, c.stoch, c.classes)).astype(np.float32); z0 /= z0.sum(-1, keepdims=True)
    z = ((1 - UNIMIX) * z0 + UNIMIX / c.classes).reshape(K, c.S).astype(np.float32)
    a = rng.uniform(-1, 1, (K, c.act)).astype(np.float32)
    obs = rng.normal(size=(H, K, c.obs)).astype(np.float32)
    return h, z, a, obs


# ----------------------------------------------------------------------------- prepare (torch)
def prepare(model: str, K: int, H: int, out: Path, seed: int = 0) -> Path:
    import torch
    sys.path.insert(0, str(WM_REPO / "wm"))
    from rssm import RSSM, RSSMConfig
    c = CFGS[model]
    ckpt = WM_REPO / "models" / f"rssm_{model}_trained" / "rssm_state_dict.pt"
    m = RSSM(RSSMConfig(**asdict(c)))
    if ckpt.exists():
        m.load_state_dict(torch.load(ckpt, map_location="cpu")); src = str(ckpt)
    else:
        m.load_state_dict({k: torch.from_numpy(v) for k, v in random_sd(c).items()}); src = "random (seed 0)"
    m.eval()
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in m.state_dict().items()}
    h, z, a, obs = initial_states(c, K, H, seed)
    T = torch.from_numpy
    with torch.no_grad():
        hs, zs, as_ = [], [], []
        ht, zt, at = T(h), T(z), T(a)
        for t in range(H):                       # imagination: prior + actor on the new feature
            ht, zt = m.prior(ht, zt, at)
            feat = torch.cat([zt, ht], -1)
            at = torch.tanh(m.actor_out(m.actor(feat))[:, :c.act])
            hs.append(ht.numpy()); zs.append(zt.numpy()); as_.append(at.numpy())
        ph, pz, pa = [], [], []
        ht, zt, at = T(h), T(z), T(a)
        for t in range(H):                       # posterior (RSSM.step) on the H observations
            ht, zt, at = m.step(T(obs[t]), ht, zt, at)
            ph.append(ht.numpy()); pz.append(zt.numpy()); pa.append(at.numpy())
    arrays = {f"sd_{k}": v for k, v in sd.items()}
    arrays.update(h0=h, z0=z, a0=a, obs=obs, ref_hs=np.stack(hs), ref_zs=np.stack(zs), ref_as=np.stack(as_),
                  ref_post_hs=np.stack(ph), ref_post_zs=np.stack(pz), ref_post_as=np.stack(pa),
                  cfg=np.array([c.obs, c.act, c.deter, c.stoch, c.classes, c.hidden, c.units]), weights=np.array(src))
    tw = Twin(c, sd).rollout(h, z, a, H)
    print(f"[wm prepare] {model}: weights {src}; K={K} H={H}; numpy fp32 twin vs torch rollout: "
          f"h {np.abs(tw[0] - arrays['ref_hs']).max():.2e} z {np.abs(tw[1] - arrays['ref_zs']).max():.2e} a {np.abs(tw[2] - arrays['ref_as']).max():.2e}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    return out


def load(npz) -> tuple[Cfg, dict, dict]:
    d = dict(np.load(npz))
    v = [int(x) for x in d["cfg"]]
    c = Cfg(obs=v[0], act=v[1], deter=v[2], stoch=v[3], classes=v[4], hidden=v[5], units=v[6])
    sd = {k[3:]: d[k] for k in d if k.startswith("sd_")}
    return c, sd, d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare")
    pp.add_argument("--model", choices=list(CFGS), default="step")
    pp.add_argument("--K", type=int, default=128)
    pp.add_argument("--H", type=int, default=10)
    pp.add_argument("--seed", type=int, default=0)
    pp.add_argument("--out", required=True)
    a = ap.parse_args()
    prepare(a.model, a.K, a.H, Path(a.out), a.seed)


if __name__ == "__main__":
    main()
