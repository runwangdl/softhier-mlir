"""Error attribution on the host (fp16-floor twins): prefix twin fed with (a) lerobot's connector output, (b) the device's
connector output (IMG hand-over); each KV through the expert twin; action error vs lerobot."""
import math
import sys
import numpy as np
from softhier_mlir.frontend import smolvla as V, smolvla_e2e as E, smolvla_expert as XE

c = sys.argv[1] if len(sys.argv) > 1 else "1"
e = dict(np.load(f'/app/models/smolvla_base/e2e_c{c}_t256.npz'))
co = np.load(f'tests/gvsoc/smolvla_e2e_app/c{c}_t256_trace/chain_outputs.npz')
w = np.load('/app/models/smolvla_base/vlm_c1.npz')
it = E.info(e)
n, n_img = it['n'], it['n_img']
lay = V.prefix_layout(it['cams'], e['lang_mask'], it['img_tok'])
r16 = lambda a: a.astype(np.float16).astype(np.float32)
f = lambda k: w[k].astype(np.float32)
tok = lay['tok'].astype(np.uint32)
allowed = (tok[None, :] <= tok[:, None]) & (tok[:, None] != V.TOK_PAD)
tab = e['p_rope'].astype(np.float32)


def prefix_kv(img):
    x = np.zeros((n, 960), np.float32)
    x[:n_img] = r16(img); x[n_img:n_img + 48] = r16(e['p_lang'].astype(np.float32))
    x[:n_img + 48] = r16(x[:n_img + 48] * np.float32(math.sqrt(960))); x[-1] = r16(e['p_state_emb'][0].astype(np.float32))
    rms = lambda a, g: a / np.sqrt((a * a).mean(1, keepdims=True) + 1e-5) * g
    kv = {}
    for L in range(16):
        ln = r16(rms(x, f(f'p_g1{L}')))
        q = r16(ln @ f(f'p_wq{L}')); k = r16(ln @ f(f'p_wk{L}')); v = r16(ln @ f(f'p_wv{L}'))
        q = r16(V.apply_rope(q.reshape(n, 15, 64), tab)).reshape(n, 960); k = r16(V.apply_rope(k.reshape(n, 5, 64), tab)).reshape(n, 320)
        kv[L] = (k.astype(np.float16), v.astype(np.float16))
        o = np.zeros((n, 960), np.float32)
        for h in range(15):
            g = h // 3
            s = r16(q[:, h * 64:(h + 1) * 64] @ k[:, g * 64:(g + 1) * 64].T) / 8
            s = np.where(allowed, s, -np.inf); m = np.where(allowed.any(1, keepdims=True), s.max(1, keepdims=True), 0)
            p = np.where(allowed, np.exp(s - m), 0); ss = p.sum(1, keepdims=True)
            p = np.where(ss > 0, p / np.where(ss > 0, ss, 1), 1 / n)
            o[:, h * 64:(h + 1) * 64] = r16(r16(p) @ v[:, g * 64:(g + 1) * 64])
        h1 = r16(x + r16(o @ f(f'p_wo{L}')))
        ln2 = r16(rms(h1, f(f'p_g2{L}')))
        gt = r16(ln2 @ f(f'p_wg{L}')); up = r16(ln2 @ f(f'p_wu{L}'))
        x = r16(h1 + r16(r16(gt / (1 + np.exp(-gt)) * up) @ f(f'p_wd{L}')))
    return kv


ref_a = e['ref_xt'][10]
valid = lay['pad']
for name, img in (("lerobot connector output", np.concatenate([e[f'ref_img_emb_{k}'] for k in range(it['cams'])])),
                  ("device connector output (IMG)", co['x_img'].astype(np.float32))):
    kv = prefix_kv(img)
    kerr = max(np.abs(kv[L][0].astype(np.float32) - e[f'ref_kv_k_{L}'])[valid].max() for L in range(16))
    d = E.expert_data(e, kv)
    P = {k[2:]: d[k] for k in d if k.startswith('p_')}
    xt, _ = XE.np_flow(P, 10, 16)
    print(f"{name}: prefix twin K max err {kerr:.3f}; expert twin actions vs lerobot max abs {np.abs(xt[10] - ref_a).max():.4f}")
d = E.expert_data(e, {L: (e[f'ref_kv_k_{L}'], e[f'ref_kv_v_{L}']) for L in range(16)})
P = {k[2:]: d[k] for k in d if k.startswith('p_')}
xt, _ = XE.np_flow(P, 10, 16)
print(f"lerobot KV (fp16-rounded): expert twin actions vs lerobot max abs {np.abs(xt[10] - ref_a).max():.4f}")
print(f"device chain actions vs lerobot: {np.abs(co['actions'] - ref_a).max():.4f}")
