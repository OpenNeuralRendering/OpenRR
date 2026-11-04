# SPDX-License-Identifier: MIT
"""Float model of gpu/shaders/include/onr_block.glsl on a case folder: fp32 products with the bias as the
initial accumulator, fp16 at the same points as the shader. Compares with the capture (exp.u8), ref-impl
(ref.u8) and the GPU output (gpu_out.u8, written by the harness with ONR_ENC_DEBUG=1).

--k8: products accumulated like the sm_86 HMMA chain instead: K in the GPU fragment order, every 8 terms
summed exactly and rounded to fp16 (RN) into an fp16 accumulator (no 2^-24 truncation). Estimates what an
fp16-accumulator WMMA path fed 8 real K per step could reach.

Usage: python gpu/tools/enc_model.py out/gpu-cases/enc3_f10 [--k8] [qv8]"""
import os, sys
import numpy as np

F16, F32, F64 = np.float16, np.float32, np.float64
GRANULE = np.array([[0, 16, 2, 18][kk // 8] + 4 * (kk % 8 // 2) + kk % 2 for kk in range(32)])


def e4m3_in(b):
    b = np.asarray(b, np.uint16)
    bits = ((b & 0x80) << 8) | ((((b >> 3) & 15) + 8) << 10) | ((b & 7) << 7)
    return bits.astype(np.uint16).view(F16)


def e4m3_out(x):
    x = np.asarray(x, F16)
    b = x.view(np.uint16).astype(np.uint32)
    r = (x.astype(F32) + (b & 0xFC40).astype(np.uint16).view(F16).astype(F32)).astype(F16)
    r = (r.astype(F32) - (b & 0xFC00).astype(np.uint16).view(F16).astype(F32)).astype(F16)
    m = np.minimum(np.maximum(np.abs(r), F16(0.00782012939453125)), F16(448))
    return ((((m.view(np.uint16).astype(np.uint32) >> 7) - 0x40) & 0x7F) | ((b >> 8) & 0x80)).astype(np.uint8)


def kv(d):
    r = {}
    for line in open(os.path.join(d, "case.txt")):
        t = line.split()
        if t:
            r[t[0]] = [int(v, 0) for v in t[1:]]
    return r


def mm_f32(a, w, bias=None):
    """a (..., K) @ w[N][K]^T in fp32 starting from the bias, one rounding to fp16."""
    acc = np.zeros(a.shape[:-1] + (w.shape[0],), F32) if bias is None else np.broadcast_to(bias.astype(F32), a.shape[:-1] + (w.shape[0],)).copy()
    return (acc + a.astype(F32) @ w.T.astype(F32)).astype(F16)


def mm_k8(a, w, bias=None, korder=None):
    """HMMA-like chain: K in korder, 8 terms per step summed exactly (fp64), fp16 accumulator rounded RN."""
    if korder is not None:
        a, w = a[..., korder], w[:, korder]
    acc = np.zeros(a.shape[:-1] + (w.shape[0],), F16) if bias is None else np.broadcast_to(np.asarray(bias, F16), a.shape[:-1] + (w.shape[0],)).copy()
    a64, w64 = a.astype(F64), w.astype(F64)
    for k in range(0, a.shape[-1], 8):
        acc = (acc.astype(F64) + a64[..., k:k + 8] @ w64[:, k:k + 8].T).astype(F16)
    return acc


def main(d, k8=False, debug=0):
    k = kv(d)
    W, H = k["size"]; C, X = k["dims"]; sx, sy = k["shift"]
    ls1, ls2 = np.array(k["ls"], np.uint16).view(F16)
    w = lambda n: np.fromfile(os.path.join(d, "w_%s.f16" % n), F16)
    in_a, in_b = w("attn_in_a").reshape(X, C), w("attn_in_b").reshape(X, C)
    wout, out_b = w("attn_out").reshape(C, X), w("attn_out_b")
    fc1, fc1_b = w("mlp_fc1").reshape(4 * C, C), w("mlp_fc1_b")
    fc2, fc2_b = w("mlp_fc2").reshape(C, 4 * C), w("mlp_fc2_b")
    bias = w("attn_bias").reshape(X // 32, 64, 64)
    gk = np.concatenate([32 * b + GRANULE for b in range(C // 32)])    # GPU K order of the residual stream
    if k8:
        mm = lambda a, ww, b=None, residual_k=False: mm_k8(a, ww, b, gk if residual_k else None)
    else:
        mm = lambda a, ww, b=None, residual_k=False: mm_f32(a, ww, b)
    x = e4m3_in(np.fromfile(os.path.join(d, "x.u8"), np.uint8)).reshape(C // 32, H, W, 32).transpose(1, 2, 0, 3).reshape(H, W, C)
    nwx, nwy = (W + sx + 7) // 8, (H + sy + 7) // 8
    ref = lambda i, n: np.minimum(np.abs(i), 2 * n - 2 - np.abs(i))
    ys = ref(np.arange(nwy * 8) - sy, H); xs = ref(np.arange(nwx * 8) - sx, W)
    t = x[ys][:, xs].reshape(nwy, 8, nwx, 8, C).transpose(0, 2, 1, 3, 4).reshape(nwy * nwx, 64, C)
    q = mm(t, in_a, residual_k=True)
    v = mm(t, in_b, residual_k=True)
    o = np.empty_like(q)
    for h in range(X // 32):
        qh = q[..., 32 * h:32 * h + 32]
        if k8:
            s = np.stack([mm_k8(qh[i], qh[i], bias[h]) for i in range(qh.shape[0])])
        else:
            s = (bias[h].astype(F32) + qh.astype(F32) @ np.swapaxes(qh, -1, -2).astype(F32)).astype(F16)
        z = np.clip((s * F16(0.017181396484375)).astype(F16), -F16(0.55615234375), F16(0.55615234375)).astype(F16)
        tt = (z.astype(F64) * -z.astype(F64) + 0.92724609375).astype(F16)
        yv = (z.astype(F64) * tt.astype(F64) + 1.375).astype(F16)
        num = ((yv.view(np.uint16).astype(np.uint32) << 5) & 0x7FE0).astype(np.uint16).view(F16)
        # row sum in the kernel's fp16 order
        tq = num.reshape(num.shape[:-1] + (8, 8))
        add = lambda a, b: (a.astype(F32) + b.astype(F32)).astype(F16)
        vv = add(add(add(tq[..., 0, :], tq[..., 2, :]), add(tq[..., 4, :], tq[..., 6, :])),
                 add(add(tq[..., 1, :], tq[..., 3, :]), add(tq[..., 5, :], tq[..., 7, :])))
        vv = vv.reshape(vv.shape[:-1] + (4, 2))
        b1 = add(vv, vv[..., [1, 0, 3, 2], :])
        b2 = add(b1, b1[..., [2, 3, 0, 1], :])
        r = (F32(1) / add(b2[..., 0, 0], b2[..., 0, 1]).astype(F32)).astype(F16)
        p = (num.astype(F32) * r[..., None].astype(F32)).astype(F16)
        vh = v[..., 32 * h:32 * h + 32]
        if k8:
            o[..., 32 * h:32 * h + 32] = np.stack([mm_k8(p[i], vh[i].T) for i in range(p.shape[0])])
        else:
            o[..., 32 * h:32 * h + 32] = (p.astype(F32) @ vh.astype(F32)).astype(F16)
    a = mm(o, wout, out_b)
    x1 = (a.astype(F64) * F64(ls1) + t.astype(F64)).astype(F16)
    pre = mm(x1, fc1, fc1_b, residual_k=True)
    tc = np.clip(pre, F16(-2), F16(2)).astype(F16)
    u = (-np.abs(tc).astype(F64) * F64(F16(0.08108)) + F64(0.412109375)).astype(F16)
    g = (tc.astype(F64) * u.astype(F64) + 0.5).astype(F16)
    hdn = (pre.astype(F64) * g.astype(F64)).astype(F16)
    m = mm(hdn, fc2, fc2_b)
    x2 = (m.astype(F64) * F64(ls2) + x1.astype(F64)).astype(F16)
    full = x2.reshape(nwy, nwx, 8, 8, C).transpose(0, 2, 1, 3, 4).reshape(nwy * 8, nwx * 8, C)[sy:sy + H, sx:sx + W]
    out = e4m3_out(full).reshape(H, W, C // 32, 32).transpose(2, 0, 1, 3).reshape(-1)
    dec = lambda b: e4m3_in(b).astype(F64)
    tag = "k8 model" if k8 else "model"

    def rep(name, a, b):
        e = np.mean((dec(a) - dec(b)) ** 2)
        print("%s vs %s: e4m3 equal %.4f%%, psnr %.2f dB" % (tag, name, 100 * np.mean(a == b), 10 * np.log10(np.abs(dec(b)).max() ** 2 / max(e, 1e-30))))
    for name, f in (("capture", "exp.u8"), ("ref-impl", "ref.u8"), ("gpu", "gpu_out.u8")):
        pth = os.path.join(d, f)
        if os.path.exists(pth):
            rep(name, out, np.fromfile(pth, np.uint8))
    if debug == "qv8":
        g = np.fromfile(os.path.join(d, "gpu_qv8.f16"), F16).reshape(nwy * nwx, 64, 2 * X)
        for name, ref_s, gs in (("Q", q, g[..., :X]), ("V", v, g[..., X:])):
            eq = np.mean(gs.view(np.uint16) == ref_s.view(np.uint16))
            e = gs.astype(F64) - ref_s.astype(F64)
            print("fp8 %s vs fp16 path: bit-equal %.4f%%, max |diff| %.4g, psnr %.2f dB" % (name, 100 * eq, np.abs(e).max(), 10 * np.log10(np.abs(ref_s.astype(F64)).max() ** 2 / np.mean(e ** 2))))


if __name__ == "__main__":
    args = sys.argv[1:]
    k8 = "--k8" in args
    args = [a for a in args if a != "--k8"]
    main(args[0], k8, args[1] if len(args) > 1 else 0)
