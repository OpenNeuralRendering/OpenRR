# SPDX-License-Identifier: MIT
"""Turn the stage B/C captures into flat case folders for the GPU harness (out/gpu-cases, never in git).

Each case folder holds raw little-endian arrays (*.f16, *.f32, *.u32, *.u8) and a case.txt with one
`key values...` line per parameter; floats are written as hex bit patterns so nothing is lost.
Expected outputs come from the capture (exp_*), ref-impl outputs (ref_*) are added where cheap.

Usage: python gpu/tools/prep_cases.py [case ...]
  cases: clear pooled ae ds post (default), enc enc1 stage dec0 dec1 head (network, in this order), blk32
ONR_REF_MAX_WINDOWS (default 400): block cases with more windows get no ref-impl output (slow).
"""
import os, struct, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.dirname(os.path.dirname(HERE))
REF = os.environ.get("OPENNR_REFIMPL", os.path.join(os.path.dirname(WT), "OpenNR-Denoiser", "ref-impl"))
sys.dont_write_bytecode = True
sys.path.insert(0, REF)
from common import *                                   # noqa: E402  (ref-impl of the main checkout, read only)
from kernels import auto_exposure as ae, downsample as ds, pooled_history2 as ph, post   # noqa: E402

OUT = os.path.join(WT, "out", "gpu-cases")


def fbits(x):
    return "0x%08x" % np.array([x], np.float32).view(np.uint32)[0]


class Case:
    def __init__(self, name):
        self.dir = os.path.join(OUT, name)
        os.makedirs(self.dir, exist_ok=True)
        self.lines = []

    def kv(self, key, *vals):
        self.lines.append(" ".join([key] + [str(v) for v in vals]))

    def arr(self, name, a):
        np.ascontiguousarray(a).tofile(os.path.join(self.dir, name))

    def close(self):
        open(os.path.join(self.dir, "case.txt"), "w").write("\n".join(self.lines) + "\n")
        print("wrote", self.dir)


def prep_clear():
    run, frame = RUNS["f0"]
    c = Case("clear_f0")
    for e in launches(run, frame):
        if e["fn"] != "rr2_clear_range_kernel":
            continue
        n = struct.unpack_from("<I", params(frame, e["idx"], run), 8)[0]
        a = e["ptrs"][0]["alloc"]
        buf = latest_dump(frame, e["idx"], "alloc", a, run)
        assert len(buf) >= n and not buf[:n].any()
        c.kv("range", e["idx"], n, len(buf))
    c.close()


def prep_pooled():
    c = Case("pooled_f10")
    src, info = c_resource(0, 4)
    _, iw = c_resource_words(0, 4)
    _, ow = c_resource_words(0, 5)
    c.kv("in", iw.shape[1], iw.shape[0])
    c.kv("out", ow.shape[1], ow.shape[0])
    c.arr("in.u32", iw.astype(np.uint32))
    c.arr("exp_out.u32", ow.astype(np.uint32))
    q = ph.quantize10(ph.run(src)).astype(np.uint32)
    c.arr("ref_out.u32", q[..., 0] | (q[..., 1] << 10) | (q[..., 2] << 20) | np.uint32(3 << 30))
    c.close()


def c_resource_words(k, rid, frame=10, scene="pan_1920x1080_quality_F"):
    import json
    base = os.path.join(CAP, "C_%s_d3d12_k%02d" % (scene, k), "trace", "d3d12_dump_f%04d_after%04d" % (frame, k))
    info = next(x for x in json.load(open(os.path.join(base, "resources.json"))) if x["id"] == rid)
    raw = np.fromfile(os.path.join(base, info["file"]), np.uint8)
    return unorm1010102(raw, info["width"], info["height"], info["row_pitch"])


def prep_ae():
    for tag, (run, frame) in RUNS.items():
        for e in launches(run, frame):
            if e["fn"] != "rr2_histogram_auto_exposure_basic_kernel":
                continue
            idx = e["idx"]
            p = params(frame, idx, run)
            x0, y0 = struct.unpack_from("<2I", p, 0)
            xmax, ymax = struct.unpack_from("<2I", p, 16)
            ncta, inv = struct.unpack_from("<If", p, 24)
            flag = struct.unpack_from("<I", p, 32)[0]
            objs = {o["off"]: o["array"] for o in e["objects"]}
            ptrs = {q["off"]: q["alloc"] for q in e["ptrs"]}
            color = np.frombuffer(latest_dump(frame, idx, "array", objs[40], run), np.float16).reshape(720, 1280, 4)
            prev = np.frombuffer(latest_dump(frame, idx, "array", objs[56], run), np.float16)[0]
            acc = latest_dump(frame, idx, "alloc", ptrs[64], run).tobytes()
            tot, nct = struct.unpack_from("<QI", acc, 0)
            exp16 = np.frombuffer(latest_dump(frame, idx, "array", objs[48], run), np.float16)[0]
            blk = np.frombuffer(latest_dump(frame, idx, "alloc", ptrs[80], run), np.float32)[:4]
            r = ae.run(color, prev, dict(x0=x0, y0=y0, xmax=xmax, ymax=ymax, ncta=ncta, inv=inv, flag=flag, grid=e["grid"][:2]))
            c = Case("ae_" + tag)
            c.kv("size", 1280, 720)
            c.kv("origin", x0, y0)
            c.kv("max", xmax, ymax)
            c.kv("grid", e["grid"][0], e["grid"][1])
            c.kv("ncta", ncta)
            c.kv("inv", fbits(inv))
            c.kv("reset", 1 if flag else 0)
            c.kv("prev", "0x%04x" % np.array([prev]).view(np.uint16)[0])
            c.kv("exp_total", tot, nct)
            c.kv("exp_exposure", "0x%04x" % np.array([exp16]).view(np.uint16)[0])
            c.kv("exp_block", *[fbits(v) for v in blk])
            c.kv("ref_total", int(r["total"]))
            c.kv("ref_exposure", "0x%04x" % np.array([r["exposure_f16"]]).view(np.uint16)[0])
            c.kv("ref_block", *[fbits(v) for v in r["block"]])
            c.arr("color.f16", color)
            c.close()


def prep_ds():
    for tag, (run, frame) in RUNS.items():
        if not os.path.isdir(run):
            continue
        c = Case("ds_" + tag)
        ls = [e for e in launches(run, frame) if e["fn"].startswith("rr2_downsample")]
        ref = None
        for k, e in enumerate(ls):
            p = ds.layout(params(frame, e["idx"], run))
            exp = np.frombuffer(latest_dump(frame, e["idx"], "alloc", 23, run), np.float32)[0]
            ptr = {q["off"]: int(q["delta"], 16) for q in e["ptrs"]}
            arena = latest_dump(frame, e["idx"], "alloc", 63, run, before=True)
            sw, sh = p["slice_size"]
            rg = arena[ptr[24]:ptr[24] + sw * sh * 4]
            b = arena[ptr[32]:ptr[32] + sw * sh * 2]
            if k == 0:
                c.kv("in_size", *p["in_size"])
                c.kv("out_size", *p["out_size"])
                c.kv("exposure", fbits(exp))
                ow, oh = p["out_size"]
                ref = np.zeros((oh, ow, 4), np.float16)
            c.kv("slice", k, *p["out_origin"], *p["out_end"], sw, sh, p["slice_x0"][0], p["slice_x1"][0])
            c.arr("rg%d.f16" % k, rg)
            c.arr("b%d.f16" % k, b)
            o = ds.to_half(ds.run(ds.slice_image(rg, b, p), p, exp))
            (a, b_), (cc, d) = p["out_origin"], p["out_end"]
            ref[b_:d, a:cc] = o
        final = latest_dump(frame, ls[-1]["idx"], "array", 9, run)
        c.arr("exp_out.f16", np.frombuffer(final, np.float16))
        c.arr("ref_out.f16", ref)
        c.close()


def prep_post(with_ref=True):
    run, frame = RUNS["f10"]
    c = Case("post_f10")
    sys.path.insert(0, REF)
    import test_post as tp
    posts = [e for e in launches(run, frame) if e["fn"] == "rr2_post_kernel"]
    for k, e in enumerate(posts):
        p, inp, ev, ptr = tp.load_inputs(run, frame, e["idx"])
        if k == 0:
            W, Hh = p["size"]
            c.kv("size", W, Hh)
            c.kv("lr_size", *p["lr_size"])
            c.kv("gate_scale", fbits(p["gate_scale"]))
            c.kv("scale", *map(fbits, p["scale"]))
            c.kv("offset", *map(fbits, p["offset"]))
            c.kv("inv_size", *map(fbits, p["inv_size"]))
            c.kv("lr_inv_size", *map(fbits, p["lr_inv_size"]))
            c.kv("reset", p["reset"])
            c.arr("dfn.f16", inp["dfn"]); c.arr("aniso.f16", inp["aniso"]); c.arr("hist.f16", inp["hist"])
            c.arr("mv.f16", inp["mv"]); c.arr("mve.f16", inp["mv_eroded"]); c.arr("mvn.f16", inp["mv_nondil"])
            c.arr("ajab.u8", inp["ajab"].astype(np.uint8)); c.arr("mvf.u8", inp["mvf"].astype(np.uint8))
            c.arr("exposure.f32", inp["exposure"].astype(np.float32))
            c.arr("bicubic.f32", post.bicubic_table().astype(np.float32))   # NVIDIA table, read from the PTX at run time
            sys.path.insert(0, HERE)
            import bicubic_table
            c.arr("bicubic_gen.f32", bicubic_table.table())                 # ours (gpu/tools/bicubic_table.py)
            ref_hist = np.zeros((Hh, W, 4), np.float16)
            ref_down = np.zeros((Hh + Hh % 2, W + W % 2, 3), np.float32)
            ref_val = np.zeros(ref_down.shape[:2], bool)
        sw, sh = p["slice_size"]
        c.kv("slice", k, *p["origin"], *p["end"], sw, sh)
        arena = latest_dump(frame, e["idx"], "alloc", 63, run)
        c.arr("exp_rg%d.f16" % k, arena[ptr[104]:ptr[104] + sw * sh * 4])
        c.arr("exp_b%d.f16" % k, arena[ptr[112]:ptr[112] + sw * sh * 2])
        if with_ref:
            t = time.time()
            r = post.run(dict(inp), p)
            xs, ys = r["xs"], r["ys"]
            ref_hist[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1] = r["hist16"]
            ref_down[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1] = r["out1"]
            ref_val[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1] = True
            o = np.zeros((sh, sw, 3), np.float16)
            nx = min(len(xs), sw)
            o[:len(ys), :nx] = r["out16"][:, :nx]
            c.arr("ref_rgb%d.f16" % k, o)
            print("  ref-impl post slice %d: %.1f s" % (k, time.time() - t))
        last = e
    surf = next(o["array"] for o in last["objects"] if o["off"] == 96)
    c.arr("exp_hist.f16", np.frombuffer(latest_dump(frame, last["idx"], "array", surf, run), np.float16))
    rid = CUDA2D3D[next(o["array"] for o in last["objects"] if o["off"] == 120)]
    _, words = c_resource_words(last["idx"], rid)
    c.kv("down_size", words.shape[1], words.shape[0])
    c.arr("exp_down.u32", words.astype(np.uint32))
    if with_ref:
        c.arr("ref_hist.f16", ref_hist)
        v, cnt = post.downsampled(ref_down, ref_val)
        q = ph.quantize10(v).astype(np.uint32)
        c.arr("ref_down.u32", q[..., 0] | (q[..., 1] << 10) | (q[..., 2] << 20) | np.uint32(3 << 30))
    c.close()


# transformer blocks whose input and TensorOutput are both in the arena: launch index -> case name.
# The weights of launch i are upload i - 1 (FASE3A). enc1 and dec*_0 merge or upsample first, enc0 embeds.
BLOCKS = {4: "enc2", 5: "enc3", 6: "enc4", 7: "enc5", 9: "dec4_1", 11: "dec3_1", 13: "dec2_1", 15: "dec1_1"}
REF_MAX_WINDOWS = int(os.environ.get("ONR_REF_MAX_WINDOWS", "400"))   # ref-impl (exact HMMA model) is slow


def write_block_case(name, x_bytes, W, Hh, L, ls, sx, sy, exp_bytes=None, with_ref=True):
    """Block case: input planes, expected planes, canonical fp16 weights; ref.u8 from ref-impl's exact model."""
    from kernels import encoder as E
    Lc = E.canonical(L)
    C, X = Lc["attn.in_a"].shape[1], Lc["attn.in_a"].shape[0]
    c = Case(name)
    c.kv("size", W, Hh)
    c.kv("dims", C, X)
    c.kv("shift", sx, sy)
    c.kv("ls", "0x%04x" % ls.view(np.uint16)[0], "0x%04x" % ls.view(np.uint16)[1])
    c.arr("x.u8", x_bytes)
    if exp_bytes is not None:
        c.arr("exp.u8", exp_bytes)
    f16 = lambda k: np.asarray(Lc[k], np.float16)
    for k in ("attn.in_a", "attn.in_b", "attn.out", "attn.out_b", "mlp.fc1", "mlp.fc1_b", "mlp.fc2", "mlp.fc2_b"):
        c.arr("w_%s.f16" % k.replace(".", "_"), f16(k))
    c.arr("w_attn_bias.f16", np.stack([E.attention_bias(L, h) for h in range(X // 32)]).astype(np.float16))
    windows = -(-(W + sx) // 8) * -(-(Hh + sy) // 8)
    if with_ref and windows <= REF_MAX_WINDOWS:
        t0 = time.time()
        x = E.read_planes(np.asarray(x_bytes, np.uint8), 0, W, Hh, C)
        y = E.run(x, L, ls[0], ls[1], sx, sy)            # exact sm_86 model, logical (GPU-order) weights
        c.arr("ref.u8", E.write_planes(y, W, Hh))
        print("  ref-impl %s: %.1f s" % (name, time.time() - t0))
    c.close()


def prep_enc(only=None, with_ref=True):
    """Every block of BLOCKS (or the ones named): TensorOutput against the capture."""
    import weights
    run, frame = RUNS["f10"]
    evs = {e["idx"]: e for e in launches(run, frame)}
    for idx, name in BLOCKS.items():
        if only and name not in only:
            continue
        ev = evs[idx]
        p = params(frame, idx, run)
        W, Hh = struct.unpack_from("<2I", p, 0)
        sx, sy = struct.unpack_from("<2H", p, 80)
        ls = np.frombuffer(p[94:98], np.float16)
        off = lambda o: int(next(q for q in ev["ptrs"] if q["off"] == o)["delta"], 16)
        before = latest_dump(frame, idx, "alloc", 63, run, before=True)
        after = latest_dump(frame, idx, "alloc", 63, run)
        L = weights.logical(idx - 1)
        n = W * Hh * L["attn.in_a"].shape[1]
        write_block_case(name + "_f10", before[off(8):off(8) + n], W, Hh, L, ls, sx, sy, after[off(32):off(32) + n], with_ref)


def _launch(idx):
    run, frame = RUNS["f10"]
    ev = next(e for e in launches(run, frame) if e["idx"] == idx)
    p = params(frame, idx, run)
    off = lambda o: int(next(q for q in ev["ptrs"] if q["off"] == o)["delta"], 16)
    before = latest_dump(frame, idx, "alloc", 63, run, before=True)
    after = latest_dump(frame, idx, "alloc", 63, run)
    W, Hh = struct.unpack_from("<2I", p, 0)
    sx, sy = struct.unpack_from("<2H", p, 80)
    return p, off, before, after, W, Hh, sx, sy


def h16(x):
    return "0x%04x" % np.asarray(x, np.float16).reshape(-1)[:1].view(np.uint16)[0]


def prep_enc1():
    """rr2_enc1: input merge (conv 2x2 stride 2 on the enc0 TensorOutput) before the block. Run `stage` after."""
    import weights
    p, off, before, after, W, Hh, sx, sy = _launch(3)
    L = weights.logical(2)
    C, Cin = L["merge_in"].shape[0], L["merge_in"].shape[1] // 4
    ls = np.frombuffer(p[94:98], np.float16)
    write_block_case("enc1_f10", before[off(8):off(8) + 4 * W * Hh * Cin], W, Hh, L, ls, sx, sy,
                     after[off(32):off(32) + W * Hh * C], with_ref=False)
    d = os.path.join(OUT, "enc1_f10")
    np.asarray(L["merge_in"], np.float16).tofile(os.path.join(d, "w_merge_in.f16"))
    np.asarray(L["merge_in_b"], np.float16).tofile(os.path.join(d, "w_merge_in_b.f16"))
    with open(os.path.join(d, "case.txt"), "a") as f:
        f.write("merge_in %d %d %d\n" % (2 * W, 2 * Hh, Cin))


DEC0 = {8: "dec4_0", 10: "dec3_0", 12: "dec2_0", 14: "dec1_0", 16: "dec0_0"}


def prep_dec0():
    """rr2_decN_kernel_0: x.u8 is the skip (encoder TensorOutput of the level), xlo.u8 the tensor of the level
    below; up = Linear + PixelShuffle 2x (logical weights [4][C][Cp]), z = fma(skip, s, up), then the block."""
    import weights
    for idx, name in DEC0.items():
        p, off, before, after, W, Hh, sx, sy = _launch(idx)
        L = weights.logical(idx - 1)
        C, Cp = L["up"].shape[1], L["up"].shape[2]
        ls = np.frombuffer(p[94:98], np.float16)
        n = W * Hh * C
        write_block_case(name + "_f10", before[off(16):off(16) + n], W, Hh, L, ls, sx, sy, after[off(32):off(32) + n], with_ref=False)
        d = os.path.join(OUT, name + "_f10")
        before[off(8):off(8) + (W // 2) * (Hh // 2) * Cp].tofile(os.path.join(d, "xlo.u8"))
        np.asarray(L["up"], np.float16).tofile(os.path.join(d, "w_up.f16"))
        np.asarray(L["up_b"], np.float16).tofile(os.path.join(d, "w_up_b.f16"))
        with open(os.path.join(d, "case.txt"), "a") as f:
            f.write("dec0 %d %s\n" % (Cp, h16(np.frombuffer(p[98:100], np.float16))))


def _dfn_w(L, rows=16):
    C = L["attn.in_a"].shape[1]
    dw, db = np.zeros((rows, C), np.float16), np.zeros(rows, np.float16)
    if "dfn" in L:
        dw[:len(L["dfn"])], db[:len(L["dfn"])] = L["dfn"], L["dfn_b"]
    return dw, db


def prep_dec1():
    """Adds the decoder DFN head to the dec4_1..dec1_1 block cases (run `enc` first): logical DFN weights, the DFN
    image of the level below (bilinear 2x), the encoder DFN image of the level (skip), the captured DFNOutput."""
    import weights
    for idx, name in BLOCKS.items():
        if not name.startswith("dec"):
            continue
        p, off, before, after, W, Hh, sx, sy = _launch(idx)
        d = os.path.join(OUT, name + "_f10")
        dw, db = _dfn_w(weights.logical(idx - 1))
        dw.tofile(os.path.join(d, "w_dfn.f16"))
        db.tofile(os.path.join(d, "w_dfn_b.f16"))
        before[off(48):off(48) + (W // 2) * (Hh // 2) * 8].tofile(os.path.join(d, "below.f16"))
        before[off(56):off(56) + W * Hh * 8].tofile(os.path.join(d, "skip.f16"))
        after[off(64):off(64) + W * Hh * 8].tofile(os.path.join(d, "exp_dfn.f16"))
        with open(os.path.join(d, "case.txt"), "a") as f:
            f.write("dec1 %s %d\n" % (h16(np.frombuffer(p[92:94], np.float16)), Hh))
        print("decoder DFN data added to", d)


def prep_head():
    """rr2_dec0_kernel_1: C = 32 block on dec0_0's output, then the head (DFN, AjabGateOgate, Aniso, MvecFactors,
    HiddenHistory) on the first 720 rows. Captured outputs: DFNOutput, Aniso (CUDA), the 8-bit surfaces (D3D12)."""
    import weights
    p, off, before, after, W, Hh, sx, sy = _launch(17)
    run, frame = RUNS["f10"]
    L = weights.logical(16)
    ls = np.frombuffer(p[94:98], np.float16)
    write_block_case("dec0_1_f10", before[off(8):off(8) + W * Hh * 32], W, Hh, L, ls, sx, sy, None, with_ref=False)
    d = os.path.join(OUT, "dec0_1_f10")
    before[off(48):off(48) + (W // 2) * (Hh // 2) * 8].tofile(os.path.join(d, "below.f16"))
    before[off(56):off(56) + W * 720 * 8].tofile(os.path.join(d, "skip.f16"))
    after[off(64):off(64) + W * 720 * 8].tofile(os.path.join(d, "exp_dfn.f16"))
    np.frombuffer(latest_dump(frame, 17, "array", 60, run).tobytes(), np.float16).tofile(os.path.join(d, "exp_aniso.f16"))
    for rid, nm in ((6, "hidden"), (48, "ajab"), (49, "mvf")):
        a, _ = c_resource(17, rid)
        np.ascontiguousarray(a).view(np.uint8).tofile(os.path.join(d, "exp_%s.u8" % nm))
    ks, kt = struct.unpack_from("<2I", p, 208)
    with open(os.path.join(d, "case.txt"), "a") as f:
        f.write("head %s 0x%08x 0x%08x 720\n" % (h16(np.frombuffer(p[92:94], np.float16)), ks, kt))


def prep_stage():
    """Adds the rest of the rr2_encN kernel to the enc1..enc5 block cases (run `enc` and `enc1` first): logical DFN
    and merge weights, the DFN image of the level and the captured PooledOutput / DFNOutput / DFNPooledOutput."""
    import weights
    run, frame = RUNS["f10"]
    evs = {e["idx"]: e for e in launches(run, frame)}
    for idx, name in {3: "enc1", **BLOCKS}.items():
        if not name.startswith("enc"):
            continue
        d = os.path.join(OUT, name + "_f10")
        ev = evs[idx]
        p = params(frame, idx, run)
        W, Hh = struct.unpack_from("<2I", p, 0)
        s_dfn = np.frombuffer(p[92:94], np.float16)
        off = lambda o: int(next(q for q in ev["ptrs"] if q["off"] == o)["delta"], 16)
        before = latest_dump(frame, idx, "alloc", 63, run, before=True)
        after = latest_dump(frame, idx, "alloc", 63, run)
        L = weights.logical(idx - 1)
        C = L["attn.in_a"].shape[1]
        dw = np.zeros((16, C), np.float16)
        dw[:3] = L["dfn"]
        db = np.zeros(16, np.float16)
        db[:3] = L["dfn_b"]
        dw.tofile(os.path.join(d, "w_stage_dfn.f16"))
        db.tofile(os.path.join(d, "w_stage_dfn_b.f16"))
        co = 0
        if "merge_out" in L:
            co = L["merge_out"].shape[0]
            np.asarray(L["merge_out"], np.float16).tofile(os.path.join(d, "w_stage_merge.f16"))
            np.asarray(L["merge_out_b"], np.float16).tofile(os.path.join(d, "w_stage_merge_b.f16"))
            n = (W // 2) * (Hh // 2) * co
            after[off(40):off(40) + n].tofile(os.path.join(d, "exp_pooled.u8"))
            after[off(72):off(72) + (W // 2) * (Hh // 2) * 8].tofile(os.path.join(d, "exp_dfn_pooled.f16"))
        before[off(48):off(48) + W * Hh * 8].tofile(os.path.join(d, "dfn_in.f16"))
        after[off(64):off(64) + W * Hh * 8].tofile(os.path.join(d, "exp_dfn.f16"))
        with open(os.path.join(d, "case.txt"), "a") as f:
            f.write("stage %d 0x%04x\n" % (co, s_dfn.view(np.uint16)[0]))
        print("stage data added to", d)


def prep_blk32(with_ref=True):
    """C = 32 block (dec0_1's weights) has no TensorOutput of its own in the capture: run it on a 128x96 crop
    of dec0_sub0 (dec0_0's TensorOutput) against ref-impl only. ls and shift as in dec1_1."""
    import weights
    from kernels import encoder as E
    run, frame = RUNS["f10"]
    evs = {e["idx"]: e for e in launches(run, frame)}
    p0 = params(frame, 16, run)
    W, Hh = struct.unpack_from("<2I", p0, 0)
    off = int(next(q for q in evs[16]["ptrs"] if q["off"] == 32)["delta"], 16)
    after = latest_dump(frame, 16, "alloc", 63, run)
    full = after[off:off + W * Hh * 32].reshape(1, Hh, W, 32)
    cw, ch = 128, 96
    crop = np.ascontiguousarray(full[:, :ch, :cw]).reshape(-1)
    p1 = params(frame, 15, run)
    sx, sy = struct.unpack_from("<2H", p1, 80)
    ls = np.frombuffer(p1[94:98], np.float16)
    write_block_case("blk32_crop", crop, cw, ch, weights.logical(16), ls, sx, sy, None, with_ref)


SCENE = os.path.join(os.path.dirname(REF), "scenes_out", "pan_1280x720_to_1920x1080_24f")
GUIDES = (("color", "color.f16"), ("diffuse_albedo", "albedo.u8"), ("specular_albedo", "spec.u8"), ("normals", "normals.f16"),
          ("roughness", "rough.f16"), ("depth", "depth.f32"), ("motion_vectors", "mv.f16"), ("specular_hit_distance", "hit.f16"))


def value_tables():
    """f32 texel values as the reference reads them: unorm10 (code / 1023), snorm8 by byte (max(c / 127, -1)), unorm8."""
    t = np.zeros(1536, np.float32)
    t[:1024] = np.arange(1024, dtype=np.float32) / np.float32(1023.0)
    t[1024:1280] = np.maximum(np.arange(256, dtype=np.uint8).view(np.int8).astype(np.float32) / np.float32(127.0), np.float32(-1))
    t[1280:] = np.arange(256, dtype=np.float32) / np.float32(255.0)
    return t


def enc0_static(c):
    """Weights of rr2_enc0: block (canonical, as the other block cases), DFN head, embed MLP (logical, padded)."""
    import weights
    from kernels import encoder as E
    L0, L1 = weights.logical(0), weights.logical(1)
    Lc = E.canonical(L1)
    for k in ("attn.in_a", "attn.in_b", "attn.out", "attn.out_b", "mlp.fc1", "mlp.fc1_b", "mlp.fc2", "mlp.fc2_b"):
        c.arr("w_%s.f16" % k.replace(".", "_"), np.asarray(Lc[k], np.float16))
    c.arr("w_attn_bias.f16", np.stack([E.attention_bias(L1, h) for h in range(2)]).astype(np.float16))
    dw, db = _dfn_w(L1)
    c.arr("w_stage_dfn.f16", dw)
    c.arr("w_stage_dfn_b.f16", db)
    w1 = np.zeros((64, 64), np.float16)
    w1[:, :43] = L0["embed.fc1"]
    c.arr("w_embed.f16", np.concatenate([w1.reshape(-1), np.asarray(L0["embed.fc2"], np.float16).reshape(-1)]))
    c.arr("w_embed_b.f16", np.concatenate([L0["embed.fc1_b"], L0["embed.fc2_b"]]).astype(np.float16))
    c.arr("lut.u16", np.asarray(L0["lut"], np.uint16))
    c.arr("tab.f32", value_tables())


def enc0_params(c, p):
    """Per-frame scalars of the rr2_enc0 launch: raw block plus the network part (shift, ls, DFN gain)."""
    assert p[0x564 - 0x160] == 0 and p[0x399 - 0x160] and p[0x56c - 0x160], "enc0 path not covered"
    assert p[0x330 - 0x160] == 0 and p[0x332 - 0x160] == 0 and p[0x39b - 0x160] == 0
    W, Hh = struct.unpack_from("<2i", p, 0)
    sx, sy = struct.unpack_from("<2H", p, 80)
    s_dfn, ls1, ls2 = np.frombuffer(p[92:98], np.uint16)
    c.kv("size", W, Hh)
    c.kv("dims", 32, 64)
    c.kv("shift", sx, sy)
    c.kv("ls", "0x%04x" % ls1, "0x%04x" % ls2)
    c.kv("stage", 0, "0x%04x" % s_dfn)
    c.kv("enc0", struct.unpack_from("<i", p, 1032)[0], p[0x331 - 0x160])
    c.arr("prm.f32", np.frombuffer(p, np.uint8))


def scene_guides(c, frame):
    for d, nm in GUIDES:
        c.arr(nm, np.fromfile(os.path.join(SCENE, d, "%04d.raw" % frame), np.uint8))


def prep_enc0(with_ref=True):
    """rr2_enc0 of frame 10: guides (scene files, identical to the dumped input arrays), PrevMvec1 / HiddenHistory /
    DownsampledHistory(2) / exposure as captured, expected arena outputs and surfaces; ref_attr.f16 = the 43 NumPy
    attribute planes before the dither."""
    run, frame = RUNS["f10"]
    c = Case("enc0_f10")
    enc0_static(c)
    p = params(frame, 2, run)
    enc0_params(c, p)
    scene_guides(c, frame)
    from enc0_inputs import input_array
    assert np.array_equal(np.fromfile(os.path.join(c.dir, "depth.f32"), np.float32), input_array(5, frame).reshape(-1))
    c.arr("pm1.f16", latest_dump(frame, 2, "array", 17, run))
    hid, _ = c_resource(0, 6, frame)
    c.arr("hidden.u8", np.ascontiguousarray(hid).view(np.uint8))
    _, w1 = c_resource_words(0, 4)
    _, w2 = c_resource_words(0, 5)
    c.kv("ds", w1.shape[1], w1.shape[0], w2.shape[1], w2.shape[0])
    c.arr("ds.u32", w1.astype(np.uint32))
    c.arr("ds2.u32", w2.astype(np.uint32))
    c.arr("expo.f32", latest_dump(frame, 2, "alloc", 23, run)[:16])
    after = latest_dump(frame, 2, "alloc", 63, run)
    W, Hh = 1280, 736
    c.arr("exp.u8", after[:W * Hh * 32])
    c.arr("exp_dfn.f16", after[0x4f10000:0x4f10000 + W * Hh * 8])
    c.arr("exp_dfn_pooled.f16", after[0x3ee4000:0x3ee4000 + W * Hh * 2])
    for nm, aid, ch in (("pmv2", 18, 4), ("ero", 25, 2), ("nond", 24, 2)):
        c.arr("exp_%s.f16" % nm, np.frombuffer(latest_dump(frame, 2, "array", aid, run).tobytes(), np.float16)[:1280 * 720 * ch])
    if with_ref:
        from kernels import enc0 as K
        I, P = K.load_inputs()
        A, surf, dfn = K.attributes(I, P)
        c.arr("ref_attr.f16", np.stack([A[k] for k in K.ORDER], -1).astype(np.float16))
    c.close()


def prep_enc0_f0():
    """rr2_enc0 of frame 0 (dump0 run): history disabled by the launch, previous-frame surfaces all zero."""
    run, frame = RUNS["f0"]
    c = Case("enc0_f0")
    enc0_static(c)
    e = next(e for e in launches(run, frame) if e["fn"] == "rr2_enc0_kernel")
    enc0_params(c, params(frame, e["idx"], run))
    scene_guides(c, frame)
    c.arr("pm1.f16", np.zeros(1280 * 720 * 4, np.float16))
    c.arr("hidden.u8", np.zeros(1280 * 720 * 4, np.uint8))
    c.kv("ds", 1440, 810, 720, 405)
    c.arr("ds.u32", np.zeros(1440 * 810, np.uint32))
    c.arr("ds2.u32", np.zeros(720 * 405, np.uint32))
    c.arr("expo.f32", latest_dump(frame, e["idx"], "alloc", 23, run)[:16])
    after = latest_dump(frame, e["idx"], "alloc", 63, run)
    W, Hh = 1280, 736
    c.arr("exp.u8", after[:W * Hh * 32])
    c.arr("exp_dfn.f16", after[0x4f10000:0x4f10000 + W * Hh * 8])
    c.arr("exp_dfn_pooled.f16", after[0x3ee4000:0x3ee4000 + W * Hh * 2])
    c.close()


def prep_seq(frames=12):
    """Frames 0..N-1 of the stage B sequence for the multi-frame harness (`seq`): per frame the guides, the rr2_enc0
    parameter block, the window shifts of every network launch, the auto-exposure / post parameters and the final
    output NGX produced. Static data (weights, tables) comes from the frame-10 cases."""
    run = RUN_B
    out_dir = os.path.join(RUN_B, "frames")
    for f in range(frames):
        c = Case(os.path.join("seq", "f%02d" % f))
        evs = launches(run, f)
        e0 = next(e for e in evs if e["fn"] == "rr2_enc0_kernel")
        enc0_params(c, params(f, e0["idx"], run))
        scene_guides(c, f)
        net = [e for e in evs if e["fn"].startswith(("rr2_enc", "rr2_dec")) and e["fn"] != "rr2_enc0_kernel"]
        c.kv("shifts", *[v for e in net for v in struct.unpack_from("<2H", params(f, e["idx"], run), 80)])
        a = next(e for e in evs if e["fn"] == "rr2_histogram_auto_exposure_basic_kernel")
        pa = params(f, a["idx"], run)
        c.kv("ae_reset", 1 if struct.unpack_from("<I", pa, 32)[0] else 0)
        if f == 0:
            r0, f0 = RUNS["f0"]
            a0 = next(e for e in launches(r0, f0) if e["fn"] == "rr2_histogram_auto_exposure_basic_kernel")
            objs = {o["off"]: o["array"] for o in a0["objects"]}
            raw = latest_dump(f0, a0["idx"], "array", objs[56], r0)
            c.kv("ae_prev", "0x%04x" % (np.frombuffer(raw, np.uint16)[0] if raw is not None else 0))
        po = next(e for e in evs if e["fn"] == "rr2_post_kernel")
        pp = post.layout(params(f, po["idx"], run))
        c.kv("post_offset", *map(fbits, pp["offset"]))
        c.kv("post_scale", *map(fbits, pp["scale"]))
        c.kv("post_reset", pp["reset"])
        c.kv("post_gate", fbits(pp["gate_scale"]))
        c.arr("exp_out.f16", np.fromfile(os.path.join(out_dir, "f%04d_output.raw" % f), np.uint8))
        c.close()


if __name__ == "__main__":
    todo = sys.argv[1:] or ["clear", "pooled", "ae", "ds", "post"]
    for t in todo:
        globals()["prep_" + t]()
