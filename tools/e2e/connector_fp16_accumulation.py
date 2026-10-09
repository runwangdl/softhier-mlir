"""Connector K=12288: fp16 running accumulation vs fp16 partials of 256 summed in fp32 (a split-K with fp32 reduction)."""
import numpy as np
from softhier_mlir.frontend import smolvla as V
e = np.load('/app/models/smolvla_base/e2e_c1_t256.npz')
vw = np.load('/app/models/smolvla_base/vision_s1024.npz')
p = {k[2:]: vw[k] for k in vw.files if k.startswith('p_')}
p['pos'] = p['pos'][e['vis_pos_ids']]
r = V.numpy_reference(p, e['xp'][0], 12)
x = V.pixel_shuffle(r['OUT'].astype(np.float16)).astype(np.float64)
wc = np.load('/app/models/smolvla_base/vlm_c1.npz')['p_wc'].astype(np.float64)
ref = e['ref_img_emb_0']
for blk in (12288, 1024, 256):
    tot = np.zeros((x.shape[0], wc.shape[1]))
    for k0 in range(0, 12288, blk):
        y = np.zeros_like(tot, dtype=np.float16)
        for k in range(k0, k0 + blk):
            y = (y.astype(np.float64) + x[:, k:k + 1] * wc[k:k + 1, :]).astype(np.float16)
        tot += y
    print(f"fp16 accumulation over {blk}, partials in fp32: max abs vs lerobot {np.abs(tot - ref).max():.4f}")
