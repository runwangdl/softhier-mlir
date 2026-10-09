"""fp8 (e4m3) quantiser of the per-step-precision flow (docs/FLOW_DATAFLOW.md, research direction R4).

Format: e4m3 as SoftHier's RedMulE model defines it (light_redmule.cpp `fp8e4m3_to_float`): sign, 4 exponent bits with
bias 7, 3 mantissa bits; exponent 15 is reserved (inf / NaN), so the largest finite value is 1.875 * 2^7 = 240;
exponent 0 is subnormal (step 2^-9). Rounding is round-to-nearest-even, out-of-range values saturate to +-240 (the
model's own float_to_fp8e4m3 clamps to inf and mis-rounds mantissas that round up to the next binade; it is not used).

Weights (host, once): per-tensor power-of-two scale s_w = 2^e_w, e_w = ceil(log2(amax / 240)); code = e4m3(w / s_w).
  Stored in HBM as one byte per element, [K, N] row-major like the fp16 copy.
  Device-side value: the byte b becomes the fp16 bit pattern ((b & 0x7F) << 7) | ((b & 0x80) << 8), which is exactly
  e4m3(b) * 2^-8 for normal and subnormal codes (both formats have the same mantissa alignment, fp16's bias is 8 more).
  The remaining factor 2^(8 + e_w) is folded into the activation (below), so the GEMM computes x_q . w_q exactly.
Activations (device, every fp8 GEMM): x_q = RN4(x) * 2^k with k = 8 + e_w, where RN4 rounds an fp16 to 4 significant
  bits by the Veltkamp split in fp16 arithmetic (c = x * 129, hi = c - (c - x): 3 fp16 SIMD ops per 4 elements). This is
  the e4m3 value of x for every |x| >= 2^-6 * s_x with a per-row power-of-two scale s_x (row max / 240 rounded up),
  i.e. for every element larger than ~1/7680 of its row maximum; smaller elements keep 4 significant bits instead of
  landing on the e4m3 subnormal grid (no per-row scale is needed for that reason: the activations here stay below 7,
  far from both 240 and fp16's limits). On every fp16 normal below 500 the split equals round-to-nearest-even to 4
  significant bits (checked exhaustively); `rn4` below reproduces the device bit for bit.
GEMM: fp16 RedMulE on (x_q, w') with fp16 accumulation and fp16 output: the numerics of an fp8-input / fp16-accumulate
  datapath. SoftHier's RedMulE fp8 mode cannot be used: it accumulates in fp8 (`matmul_fp8e4m3`: every FMA rounded to
  e4m3) and its address generator steps by the arch constant elem_size = 2, so its byte-indexed fp8 compute reads a
  scrambled tile.
"""
from __future__ import annotations

import numpy as np

E4M3_MAX = 240.0
E4M3_MIN_NORMAL = 2.0 ** -6
E4M3_SUB_STEP = 2.0 ** -9


def e4m3_round(v: np.ndarray) -> np.ndarray:
    """float64 values rounded to the e4m3 grid (RNE, saturating at +-240)."""
    v = np.asarray(v, np.float64)
    a = np.abs(v)
    out = np.empty_like(a)
    sub = a < E4M3_MIN_NORMAL
    out[sub] = np.round(a[sub] / E4M3_SUB_STEP) * E4M3_SUB_STEP          # np.round = RNE
    an = a[~sub]
    e = np.floor(np.log2(an))
    ulp = 2.0 ** (e - 3)
    out[~sub] = np.round(an / ulp) * ulp
    return np.sign(v) * np.minimum(out, E4M3_MAX)


def e4m3_encode(v: np.ndarray) -> np.ndarray:
    """values already on the e4m3 grid -> uint8 codes (sign | exponent << 3 | mantissa)."""
    v = np.asarray(v, np.float64)
    sign = (v < 0).astype(np.uint8) << 7
    a = np.abs(v)
    code = np.zeros(a.shape, np.uint8)
    sub = a < E4M3_MIN_NORMAL
    code[sub] = np.round(a[sub] / E4M3_SUB_STEP).astype(np.uint8)          # 0..8 (8 = the smallest normal, same code)
    an = a[~sub]
    e = np.floor(np.log2(an)).astype(np.int64)
    m = np.round((an / 2.0 ** e - 1.0) * 8).astype(np.int64)
    code[~sub] = (((e + 7) << 3) | m).astype(np.uint8)
    return code | sign


def e4m3_decode(code: np.ndarray) -> np.ndarray:
    """uint8 codes -> float64 values (exponent 15 is never produced by e4m3_encode)."""
    c = np.asarray(code, np.uint8).astype(np.int64)
    s = np.where(c & 0x80, -1.0, 1.0)
    e, m = (c >> 3) & 0xF, c & 7
    return s * np.where(e == 0, m * E4M3_SUB_STEP, (1 + m / 8.0) * 2.0 ** (e - 7))


def quant_weight(w: np.ndarray) -> tuple[np.ndarray, int]:
    """[K, N] weights -> (uint8 codes [K, N], e_w) with the per-tensor scale 2^e_w."""
    w = np.asarray(w, np.float64)
    amax = float(np.abs(w).max())
    e_w = int(np.ceil(np.log2(amax / E4M3_MAX))) if amax > 0 else 0
    return e4m3_encode(e4m3_round(w / 2.0 ** e_w)), e_w


def w_prime(code: np.ndarray) -> np.ndarray:
    """The fp16 the device builds from each code: bits ((b & 0x7F) << 7) | ((b & 0x80) << 8) = e4m3(b) * 2^-8."""
    b = np.asarray(code, np.uint8).astype(np.uint16)
    return (((b & 0x7F) << 7) | ((b & 0x80) << 8)).astype(np.uint16).view(np.float16)


def rn4(x16: np.ndarray, k: int = 0) -> np.ndarray:
    """The device's activation cast: Veltkamp split to 4 significant bits in fp16 arithmetic, then * 2^k (fp16)."""
    x = np.asarray(x16, np.float16)
    c = (x * np.float16(129.0)).astype(np.float16)
    d = (c - x).astype(np.float16)
    hi = (c - d).astype(np.float16)
    return (hi * np.float16(2.0 ** k)).astype(np.float16)


def gemm_ref(x16: np.ndarray, code: np.ndarray, e_w: int) -> np.ndarray:
    """fp32-accumulated reference of the device fp8 GEMM, rounded to fp16."""
    xq = rn4(x16, 8 + e_w).astype(np.float32)
    return (xq @ w_prime(code).astype(np.float32)).astype(np.float16)


def dequant(code: np.ndarray, e_w: int) -> np.ndarray:
    return e4m3_decode(code) * 2.0 ** e_w
