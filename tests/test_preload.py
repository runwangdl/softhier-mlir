"""Host-side checks of the HBM preload image and the SmolVLA weight layout (no simulator)."""
import numpy as np
import pytest

from softhier_mlir.sim.preload import HBM_BASE, SENTINEL_MAGIC, make_preload_elf, read_preload_elf, sentinel_array


def test_preload_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    arrays = {0x1000: rng.standard_normal((64, 96)).astype(np.float16),
              0x200000: np.arange(1000, dtype=np.float16).reshape(10, 100),
              (200 << 20): rng.standard_normal((3, 5)).astype(np.float16)}
    elf = make_preload_elf(tmp_path / "p.elf", arrays)
    back = read_preload_elf(elf)
    assert set(back) == set(arrays)
    for off, a in arrays.items():
        assert back[off] == a.tobytes()
    # loader-relevant header fields: ELFCLASS32, little-endian, RISC-V, one PT_LOAD per array
    raw = elf.read_bytes()
    assert raw[:4] == b"\x7fELF" and raw[4] == 1 and raw[5] == 1
    assert int.from_bytes(raw[18:20], "little") == 243
    assert int.from_bytes(raw[44:46], "little") == len(arrays)


def test_preload_rejects_allocator_region_and_overlap(tmp_path):
    with pytest.raises(ValueError):
        make_preload_elf(tmp_path / "a.elf", {0x400: np.zeros(4, np.float16)})
    with pytest.raises(ValueError):
        make_preload_elf(tmp_path / "b.elf", {0x1000: np.zeros(64, np.float16), 0x1040: np.zeros(4, np.float16)})
    with pytest.raises(ValueError):
        make_preload_elf(tmp_path / "c.elf", {0x1010: np.zeros(4, np.float16)})


def test_sentinel_is_last_segment(tmp_path):
    sent = sentinel_array()
    assert sent.shape == (1, 32) and sent.nbytes == 64 and (sent.view(np.uint32) == SENTINEL_MAGIC).all()
    arrays = {0x200000: np.zeros((8, 8), np.float16), 0x1000: np.ones((4, 4), np.float16), 0x300000: sent}
    back = read_preload_elf(make_preload_elf(tmp_path / "s.elf", arrays))
    assert list(back) == [0x1000, 0x200000, 0x300000]           # written in offset order: the sentinel lands last
    assert back[0x300000] == sent.tobytes()


def test_smolvla_emit_waits_before_first_use():
    """The emitted program must not touch preloaded data before softhier.preload_wait on the last segment."""
    from softhier_mlir.frontend import smolvla
    rng = np.random.default_rng(2)
    D, FF = smolvla.D, smolvla.FF
    P = {"wpe": (D, D), "bpe": (1, D), "pos": (16, D), "gpost": (1, D), "bepost": (1, D)}
    for nm, shp in (("wq", (D, D)), ("wk", (D, D)), ("wv", (D, D)), ("wo", (D, D)), ("w1", (D, FF)), ("w2", (FF, D)), ("bq", (1, D)), ("bk", (1, D)),
                    ("bv", (1, D)), ("bo", (1, D)), ("b1", (1, FF)), ("b2", (1, D)), ("g1", (1, D)), ("be1", (1, D)), ("g2", (1, D)), ("be2", (1, D))):
        P[nm + "0"] = shp
    data = {f"p_{k}": rng.standard_normal(v).astype(np.float16) for k, v in P.items()}
    data["xp"] = rng.standard_normal((16, D)).astype(np.float16); data["meta"] = np.array([16, 1, 0])
    mlir, pre = smolvla.emit(data, layers=1)
    last = max(pre); assert pre[last].tobytes() == sentinel_array().tobytes()
    lines = mlir.splitlines()
    first_use = next(i for i, ln in enumerate(lines) if "softhier.gemm" in ln or "softhier.dump_samples" in ln)
    assert any("softhier.preload_wait %sentinel" in ln for ln in lines[:first_use])


def test_smolvla_layout_and_im2col():
    from softhier_mlir.frontend import smolvla
    rng = np.random.default_rng(1)
    D, FF = smolvla.D, smolvla.FF
    vis = {"embeddings.patch_embedding.weight": rng.standard_normal((D, 3, 16, 16)).astype(np.float32),
           "embeddings.patch_embedding.bias": rng.standard_normal(D).astype(np.float32),
           "embeddings.position_embedding.weight": rng.standard_normal((1024, D)).astype(np.float32),
           "post_layernorm.weight": np.ones(D, np.float32), "post_layernorm.bias": np.zeros(D, np.float32)}
    for L in range(smolvla.LAYERS):
        g = f"encoder.layers.{L}."
        for nm, (o, i) in (("self_attn.q_proj", (D, D)), ("self_attn.k_proj", (D, D)), ("self_attn.v_proj", (D, D)),
                           ("self_attn.out_proj", (D, D)), ("mlp.fc1", (FF, D)), ("mlp.fc2", (D, FF))):
            vis[g + nm + ".weight"] = rng.standard_normal((o, i)).astype(np.float32)
            vis[g + nm + ".bias"] = rng.standard_normal(o).astype(np.float32)
        for nm in ("layer_norm1", "layer_norm2"):
            vis[g + nm + ".weight"] = np.ones(D, np.float32); vis[g + nm + ".bias"] = np.zeros(D, np.float32)
    assert len(vis) == 197
    ids = smolvla.token_ids_for(256)
    assert ids[:3].tolist() == [0, 1, 2] and ids[16] == 32          # top-left 16x16 patches of the 32x32 grid
    p = smolvla.to_library_layout(vis, ids)
    assert p["w10"].shape == (D, FF) and p["w20"].shape == (FF, D) and p["b10"].shape == (1, FF)
    assert np.array_equal(p["wq3"].astype(np.float32), vis["encoder.layers.3.self_attn.q_proj.weight"].T.astype(np.float16).astype(np.float32))
    assert p["pos"].shape == (256, D) and np.array_equal(p["pos"][17], vis["embeddings.position_embedding.weight"][33].astype(np.float16))
    # im2col(pixels) @ wpe == the 16x16 stride-16 convolution
    pixels = rng.standard_normal((3, 512, 512)).astype(np.float32)
    xp = smolvla.im2col(pixels, ids)
    t = 5 * 16 + 7
    patch = pixels[:, 5 * 16:6 * 16, 7 * 16:8 * 16]
    want = np.tensordot(vis["embeddings.patch_embedding.weight"], patch, axes=([1, 2, 3], [0, 1, 2]))
    got = xp[t] @ vis["embeddings.patch_embedding.weight"].reshape(D, -1).T
    assert np.allclose(got, want, atol=1e-3)
