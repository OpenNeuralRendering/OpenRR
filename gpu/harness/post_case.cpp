// SPDX-License-Identifier: MIT
// rr2_post on frame 10: four slices, then our slices through our downsample against the final output.
// Tolerance case (docs/FASE3C.md section 4): the residual is exp2/log2 rounding, as in ref-impl.
#include <cstdio>
#include <string>

#include "harness.hpp"

using namespace onr;

template <class T> static Buffer make(Ctx& c, const std::vector<T>& v) {
    Buffer b = c.dev.create(v.size() * sizeof(T));
    c.dev.upload(b, v.data(), v.size() * sizeof(T));
    return b;
}

template <class T> static std::vector<T> fetch(Ctx& c, const Buffer& b, size_t n) {
    std::vector<T> v(n);
    c.dev.download(b, v.data(), n * sizeof(T));
    return v;
}

int run_post(Ctx& c, const CaseFile& cf) {
    I2 size{cf.i("size", 0), cf.i("size", 1)}, lr{cf.i("lr_size", 0), cf.i("lr_size", 1)};
    I2 dsz{cf.i("down_size", 0), cf.i("down_size", 1)};
    std::vector<Buffer> all;
    auto keep = [&](Buffer b) { all.push_back(b); return b; };
    Buffer dfn = keep(make(c, cf.read<uint16_t>("dfn.f16"))), ajab = keep(make(c, cf.read<uint8_t>("ajab.u8")));
    Buffer aniso = keep(make(c, cf.read<uint16_t>("aniso.f16"))), mvf = keep(make(c, cf.read<uint8_t>("mvf.u8")));
    Buffer hist = keep(make(c, cf.read<uint16_t>("hist.f16"))), mv = keep(make(c, cf.read<uint16_t>("mv.f16")));
    Buffer mve = keep(make(c, cf.read<uint16_t>("mve.f16"))), mvn = keep(make(c, cf.read<uint16_t>("mvn.f16")));
    Buffer expo = keep(make(c, cf.read<float>("exposure.f32"))), bic = keep(make(c, cf.read<float>("bicubic.f32")));
    Buffer tile = keep(c.dev.create(size_t(lr.x) * lr.y * 24));
    const size_t hist_n = size_t(size.x) * size.y * 4;
    Buffer hout = keep(c.dev.create(hist_n * 2)), down = keep(c.dev.create(size_t(dsz.x) * dsz.y * 4));
    Buffer fix = keep(c.dev.create(post_fix_bytes(size)));
    c.dev.fill(hout, 0);
    c.dev.fill(down, 0);

    PostTilePC tp{dfn.addr, ajab.addr, aniso.addr, tile.addr, lr, lr.x};
    std::vector<PostPC> ps;
    std::vector<std::pair<Buffer, Buffer>> outs;
    auto slices = cf.all("slice");
    for (auto& s : slices) {
        auto v = [&](int i) { return std::stoi(s[i]); };
        I2 sl{v(5), v(6)};
        Buffer rg = keep(c.dev.create(size_t(sl.x) * sl.y * 4)), b = keep(c.dev.create(size_t(sl.x) * sl.y * 2));
        c.dev.fill(rg, 0);
        c.dev.fill(b, 0);
        outs.push_back({rg, b});
        PostPC p{};
        p.tile = tile.addr; p.mvf = mvf.addr; p.hist_in = hist.addr; p.mv = mv.addr; p.mv_eroded = mve.addr;
        p.mv_nondil = mvn.addr; p.exposure = expo.addr; p.bicubic = bic.addr; p.hist_out = hout.addr;
        p.out_rg = rg.addr; p.out_b = b.addr; p.down = down.addr; p.fix = fix.addr;
        p.gate_scale = cf.f("gate_scale");
        for (int i = 0; i < 2; ++i) {
            p.scale[i] = cf.f("scale", i); p.offset[i] = cf.f("offset", i);
            p.inv_size[i] = cf.f("inv_size", i); p.lr_inv_size[i] = cf.f("lr_inv_size", i);
        }
        p.size = size; p.lr_size = lr; p.origin = {v(1), v(2)}; p.end = {v(3), v(4)}; p.slice = sl;
        p.reset = uint32_t(cf.i("reset"));
        ps.push_back(p);
    }
    auto record = [&] {
        c.k.post_tile(c.dev, tp);
        c.dev.barrier();
        for (auto& p : ps) { c.k.post(c.dev, p, fix); c.dev.barrier(); }
    };
    c.dev.begin();
    record();
    auto t0 = c.dev.submit();
    double ms = time_it(c, record);

    int bad = 0;
    auto h = fetch<uint16_t>(c, hout, hist_n);
    bad += cmp_f16("InternalHistory rgba vs capture", h, cf.read<uint16_t>("exp_hist.f16")) < 0.9997;
    cmp_f16("InternalHistory rgba vs ref-impl", h, cf.read<uint16_t>("ref_hist.f16"));
    for (size_t k = 0; k < slices.size(); ++k) {
        I2 sl = ps[k].slice;
        size_t n = size_t(sl.x) * sl.y;
        auto rg = fetch<uint16_t>(c, outs[k].first, n * 2), b = fetch<uint16_t>(c, outs[k].second, n);
        auto erg = cf.read<uint16_t>("exp_rg" + std::to_string(k) + ".f16");
        auto eb = cf.read<uint16_t>("exp_b" + std::to_string(k) + ".f16");
        auto ref = cf.read<uint16_t>("ref_rgb" + std::to_string(k) + ".f16");
        // only the columns this launch writes (the capture keeps stale arena bytes in the rest)
        const int wx = std::min(ps[k].end.x, size.x) - ps[k].origin.x;
        std::vector<uint16_t> ours, capt, refw;
        for (size_t i = 0; i < n; ++i) {
            if (int(i % size_t(sl.x)) >= wx) continue;
            for (int q = 0; q < 3; ++q) {
                ours.push_back(q < 2 ? rg[2 * i + q] : b[i]);
                capt.push_back(q < 2 ? erg[2 * i + q] : eb[i]);
                refw.push_back(ref[3 * i + q]);
            }
        }
        std::string w = "OutputHistRes slice " + std::to_string(k) + " vs capture";
        bad += cmp_f16(w.c_str(), ours, capt) < 0.9996;
        w = "OutputHistRes slice " + std::to_string(k) + " vs ref-impl";
        cmp_f16(w.c_str(), ours, refw);
    }
    auto dh = fetch<uint32_t>(c, down, size_t(dsz.x) * dsz.y);
    bad += cmp_1010102("DownsampledHistory vs capture", dh, cf.read<uint32_t>("exp_down.u32")) < 0.9999;
    cmp_1010102("DownsampledHistory vs ref-impl", dh, cf.read<uint32_t>("ref_down.u32"));
    printf("  gpu %.4f ms (tile + 4 slices; first run %.4f ms)\n", ms, [&] { double s = 0; for (double x : t0) s += x; return s; }());

    // chain: our OutputHistRes slices -> downsample -> final output (frame 10)
    CaseFile ds;
    if (ds.load(cf.dir.parent_path() / "ds_f10")) {
        I2 in{ds.i("in_size", 0), ds.i("in_size", 1)}, out{ds.i("out_size", 0), ds.i("out_size", 1)};
        Buffer dst = keep(c.dev.create(size_t(out.x) * out.y * 8));
        c.dev.begin();
        size_t k = 0;
        for (auto& s : ds.all("slice")) {
            auto v = [&](int i) { return std::stoi(s[i]); };
            DownsamplePC p{outs[k].first.addr, outs[k].second.addr, dst.addr, expo.addr, in, out,
                           {v(1), v(2)}, {v(3), v(4)}, {v(5), v(6)}, {v(7), v(8)}};
            c.k.downsample(c.dev, p);
            ++k;
        }
        c.dev.submit();
        auto o = fetch<uint16_t>(c, dst, size_t(out.x) * out.y * 4);
        bad += cmp_f16("chain post->downsample, output RGB vs capture", o, ds.read<uint16_t>("exp_out.f16"), 4, 3) < 0.9995;

        // whole frame: one post dispatch, one downsample dispatch (no slice margins)
        Buffer frg = keep(c.dev.create(size_t(size.x) * size.y * 4)), fb = keep(c.dev.create(size_t(size.x) * size.y * 2));
        Buffer hout2 = keep(c.dev.create(hist_n * 2)), down2 = keep(c.dev.create(size_t(dsz.x) * dsz.y * 4));
        Buffer dst2 = keep(c.dev.create(size_t(out.x) * out.y * 8));
        PostPC p = ps[0];
        p.hist_out = hout2.addr; p.out_rg = frg.addr; p.out_b = fb.addr; p.down = down2.addr;
        p.origin = {0, 0}; p.end = size; p.slice = size;
        DownsamplePC dp{frg.addr, fb.addr, dst2.addr, expo.addr, in, out, {0, 0}, out, size, {0, size.x}};
        auto rec_full = [&] {
            c.k.post_tile(c.dev, tp);
            c.dev.barrier();
            c.k.post(c.dev, p, fix);
            c.dev.barrier();
            c.k.downsample(c.dev, dp);
        };
        c.dev.begin();
        rec_full();
        c.dev.submit();
        double fms = time_it(c, rec_full);
        auto fx = fetch<uint32_t>(c, fix, 4);
        printf("  fixup list: %u blocks of 2x2 (%.3f%% of the blocks), %u workgroups\n", fx[3], 100.0 * fx[3] / (double(dsz.x) * dsz.y), fx[0]);
        bad += cmp_f16("whole frame: InternalHistory vs capture", fetch<uint16_t>(c, hout2, hist_n), cf.read<uint16_t>("exp_hist.f16")) < 0.9997;
        bad += cmp_1010102("whole frame: DownsampledHistory vs capture", fetch<uint32_t>(c, down2, size_t(dsz.x) * dsz.y), cf.read<uint32_t>("exp_down.u32")) < 0.9999;
        auto o2 = fetch<uint16_t>(c, dst2, size_t(out.x) * out.y * 4);
        bad += cmp_f16("whole frame: output RGB vs capture", o2, ds.read<uint16_t>("exp_out.f16"), 4, 3) < 0.9995;
        printf("  gpu %.4f ms (whole frame: tile + post + downsample)\n", fms);

        // fixup pass check: every block listed and recomputed with the exact blend must give the same bits
        {
            auto h1 = fetch<uint16_t>(c, hout2, hist_n);
            auto o1 = fetch<uint16_t>(c, dst2, size_t(out.x) * out.y * 4);
            auto d1 = fetch<uint32_t>(c, down2, size_t(dsz.x) * dsz.y);
            p.reset |= 2u;
            c.dev.begin();
            rec_full();
            auto t = c.dev.submit();
            p.reset &= ~2u;
            auto fx2 = fetch<uint32_t>(c, fix, 4);
            bool same = h1 == fetch<uint16_t>(c, hout2, hist_n) && o1 == fetch<uint16_t>(c, dst2, size_t(out.x) * out.y * 4) &&
                        d1 == fetch<uint32_t>(c, down2, size_t(dsz.x) * dsz.y);
            printf("  fixup of all %u blocks (exact blend everywhere, %.3f ms): %s\n", fx2[3], t.size() > 2 ? t[2] : 0.0,
                   same ? "identical outputs" : "OUTPUTS DIFFER");
            bad += !same;
        }

        // the same with our generated bicubic table (gpu/tools/bicubic_table.py) instead of NVIDIA's
        if (cf.has_file("bicubic_gen.f32")) {
            Buffer bg = keep(make(c, cf.read<float>("bicubic_gen.f32")));
            p.bicubic = bg.addr;
            c.dev.begin();
            rec_full();
            c.dev.submit();
            cmp_f16("own bicubic table: InternalHistory vs capture", fetch<uint16_t>(c, hout2, hist_n), cf.read<uint16_t>("exp_hist.f16"));
            cmp_f16("own bicubic table: output RGB vs capture", fetch<uint16_t>(c, dst2, size_t(out.x) * out.y * 4), ds.read<uint16_t>("exp_out.f16"), 4, 3);
        }
    }
    for (auto& b : all) c.dev.destroy(b);
    return bad != 0;
}
