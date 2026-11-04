// SPDX-License-Identifier: MIT
// Probes of the AMD GPU's arithmetic (no capture needed): `opennr_rr_harness probe`.
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "harness.hpp"

using namespace onr;

static int probe_div(Ctx& c) {
    const uint32_t n = 1u << 24;
    Buffer a = c.dev.create(size_t(n) * 4), b = c.dev.create(size_t(n) * 4);
    struct { uint64_t a, b; } pc{a.addr, b.addr};
    c.dev.begin();
    c.dev.dispatch(c.k["probe_div"], &pc, sizeof pc, n / 256);
    c.dev.submit();
    std::vector<uint32_t> va(n), vb(n);
    c.dev.download(a, va.data(), size_t(n) * 4);
    c.dev.download(b, vb.data(), size_t(n) * 4);
    size_t ba = 0, bb = 0;
    for (uint32_t s = 0; s < n; ++s) {
        uint32_t r = fbits(float(double(s) / 255.0));
        ba += va[s] != r;
        bb += vb[s] != r;
    }
    printf("  s/255, s < 2^24: fp32 routine %zu wrong, GPU fp64 division %zu wrong\n", ba, bb);
    c.dev.destroy(a);
    c.dev.destroy(b);
    return ba != 0;
}

static uint16_t half_rn(double x) {
    uint16_t s = std::signbit(x) ? 0x8000 : 0;
    double a = std::fabs(x);
    if (a == 0) return s;
    int e;
    std::frexp(a, &e);                                  // a = f 2^e, f in [0.5, 1)
    int qe = std::max(e - 1, -14) - 10;
    double m = std::nearbyint(std::ldexp(a, -qe));      // ties to even
    double v = std::ldexp(m, qe);
    if (v >= 65520.0) return uint16_t(s | 0x7c00);
    if (v < std::ldexp(1.0, -14)) return uint16_t(s | uint16_t(std::ldexp(v, 24)));
    std::frexp(v, &e);
    return uint16_t(s | ((e - 1 + 15) << 10) | (uint16_t(std::ldexp(v, 11 - e)) & 1023));
}

// exact reference rule on the CPU: 8-bit rounded fraction, exact blend, fp16 RN
static uint16_t ref_bilinear(const std::vector<uint16_t>& img, int w, int h, float u, float v, int ch) {
    double tx = double(u) * w * 256.0 - 128.0, ty = double(v) * h * 256.0 - 128.0;
    long long qx = (long long)std::floor(tx + 0.5), qy = (long long)std::floor(ty + 0.5);
    int x0 = int(std::floor(qx / 256.0)), y0 = int(std::floor(qy / 256.0));
    int ax = int(qx - 256ll * x0), ay = int(qy - 256ll * y0);
    auto T = [&](int x, int y) {
        x = std::clamp(x, 0, w - 1); y = std::clamp(y, 0, h - 1);
        return double(to_float(img[(size_t(y) * w + x) * 4 + ch]));
    };
    double s = double((256 - ax) * (256 - ay)) * T(x0, y0) + double(ax * (256 - ay)) * T(x0 + 1, y0) +
               double((256 - ax) * ay) * T(x0, y0 + 1) + double(ax * ay) * T(x0 + 1, y0 + 1);
    return half_rn(s / 65536.0);
}

struct Lcg {
    uint64_t s = 0x9e3779b97f4a7c15ull;
    uint32_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return uint32_t(s >> 33); }
};

static uint16_t half_bits_rand(Lcg& r, bool sign) {
    uint32_t e = 1 + r.next() % 15, m = r.next() & 1023;   // 2^-14 .. 2^0
    return uint16_t(((sign && (r.next() & 1)) ? 0x8000u : 0u) | (e << 10) | m);
}

// AMD sampler vs the reference rule on an rgba16f image (the captured history, or random texels).
static int probe_tex(Ctx& c, const std::vector<uint16_t>& img, uint32_t w, uint32_t h, const char* what) {
    const uint32_t n = 1u << 22;
    Lcg r;
    std::vector<float> uv(2 * size_t(n));
    for (auto& x : uv) x = (float(r.next() >> 8) + 0.5f) / 16777216.f * 1.02f - 0.01f;
    for (size_t i = 0; i < n; i += 2) {   // every other sample on a texel centre in x or y (2-texel path)
        size_t k = 2 * i + (i & 2 ? 1 : 0);
        float sz = float(k & 1 ? h : w);
        uv[k] = (std::floor(uv[k] * sz) + 0.5f) / sz;
    }
    Buffer bimg = c.dev.create(img.size() * 2), buv = c.dev.create(uv.size() * 4);
    c.dev.upload(bimg, img.data(), img.size() * 2);
    c.dev.upload(buv, uv.data(), uv.size() * 4);
    Buffer bm = c.dev.create(size_t(n) * 8), bn = c.dev.create(size_t(n) * 8), bq = c.dev.create(size_t(n) * 8);
    Buffer braw = c.dev.create(size_t(n) * 16);
    Image im = c.dev.create_image(w, h, VK_FORMAT_R16G16B16A16_SFLOAT, bimg);
    VkSampler sn = c.dev.sampler(true, false), su = c.dev.sampler(true, true);
    c.dev.set_texture(0, im, sn);
    c.dev.set_texture(1, im, su);
    struct { uint64_t uv, img, m, nat, q, raw; int32_t w, h; uint32_t n; uint32_t mode; } pc{buv.addr, bimg.addr, bm.addr, bn.addr, bq.addr,
                                                                                 braw.addr, int32_t(w), int32_t(h), n, 0};
    c.dev.begin();
    c.dev.dispatch(c.k["probe_tex"], &pc, sizeof pc, n / 256);
    c.dev.submit();
    std::vector<uint16_t> m(size_t(n) * 4), nat(size_t(n) * 4), q(size_t(n) * 4);
    std::vector<float> raw(size_t(n) * 4);
    c.dev.download(bm, m.data(), m.size() * 2);
    c.dev.download(bn, nat.data(), nat.size() * 2);
    c.dev.download(bq, q.data(), q.size() * 2);
    c.dev.download(braw, raw.data(), raw.size() * 4);
    size_t dm = 0;
    for (size_t i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) dm += m[4 * i + k] != ref_bilinear(img, int(w), int(h), uv[2 * i], uv[2 * i + 1], k);
    printf("  %s: manual GPU filter vs CPU exact rule: %zu of %zu differ\n", what, dm, size_t(n) * 3);
    size_t dn = 0, dq = 0, tot = size_t(n) * 3;
    double maxrel = 0;
    for (size_t i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) {
            dn += m[4 * i + k] != nat[4 * i + k];
            dq += m[4 * i + k] != q[4 * i + k];
            double e = to_float(m[4 * i + k]), g = raw[4 * i + k];
            if (e != 0) maxrel = std::max(maxrel, std::fabs(g - e) / std::fabs(e));
        }
    printf("  %s: sampler with normalized coords differs in %zu of %zu values (%.4f%%); "
           "with coords pre-rounded to 8 bits %zu (%.4f%%); max rel. error of its fp32 output %.3g\n",
           what, dn, tot, 100.0 * double(dn) / double(tot), dq, 100.0 * double(dq) / double(tot), maxrel);
    c.dev.destroy(sn);
    c.dev.destroy(su);
    c.dev.destroy(im);
    for (Buffer* b : {&bimg, &buv, &bm, &bn, &bq, &braw}) c.dev.destroy(*b);
    return 0;
}

// Weight probe: R = x & 1, G = y & 1, so the sampler's fp32 output is its own x / y weight.
static void probe_weights(Ctx& c) {
    const uint32_t w = 64, h = 64, n = 1u << 22;
    std::vector<uint16_t> img(w * h * 4);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            uint16_t* p = &img[(y * w + x) * 4];
            p[0] = (x & 1) ? 0x3c00 : 0; p[1] = (y & 1) ? 0x3c00 : 0; p[2] = 0x3800; p[3] = 0x3c00;
        }
    Lcg r;
    std::vector<float> uv(2 * size_t(n));
    for (auto& x : uv) x = (float(r.next() >> 8) + 0.5f) / 16777216.f * 0.8f + 0.1f;
    Buffer bimg = c.dev.create(img.size() * 2), buv = c.dev.create(uv.size() * 4);
    c.dev.upload(bimg, img.data(), img.size() * 2);
    c.dev.upload(buv, uv.data(), uv.size() * 4);
    Buffer bm = c.dev.create(size_t(n) * 8), bn = c.dev.create(size_t(n) * 8), bq = c.dev.create(size_t(n) * 8);
    Buffer braw = c.dev.create(size_t(n) * 16);
    Image im = c.dev.create_image(w, h, VK_FORMAT_R16G16B16A16_SFLOAT, bimg);
    VkSampler sn = c.dev.sampler(true, false), su = c.dev.sampler(true, true);
    c.dev.set_texture(0, im, sn);
    c.dev.set_texture(1, im, sn);   // normalized for the raw output too
    struct { uint64_t uv, img, m, nat, q, raw; int32_t w, h; uint32_t n; uint32_t mode; } pc{
        buv.addr, bimg.addr, bm.addr, bn.addr, bq.addr, braw.addr, int32_t(w), int32_t(h), n, 1};
    c.dev.begin();
    c.dev.dispatch(c.k["probe_tex"], &pc, sizeof pc, n / 256);
    c.dev.submit();
    std::vector<float> raw(size_t(n) * 4);
    c.dev.download(braw, raw.data(), raw.size() * 4);
    size_t not8 = 0, rnd = 0, trn = 0;
    int hist[8] = {};
    for (size_t i = 0; i < n; ++i) {
        double u = uv[2 * i], t = u * w - 0.5;
        double fr = t - std::floor(t);
        int x0 = int(std::floor(t));
        double a = raw[4 * i];               // weight of the odd texel
        double wx = (x0 & 1) ? 1.0 - a : a;  // weight of texel x0 + 1
        double s = wx * 256.0;
        if (s != std::floor(s)) { ++not8; continue; }
        int q = int(s);
        rnd += q == int(std::floor(fr * 256.0 + 0.5)) % 256 || (q == 0 && std::floor(fr * 256.0 + 0.5) == 256);
        trn += q == int(std::floor(fr * 256.0));
        int d = q - int(std::floor(fr * 256.0));
        if (d >= -3 && d <= 4) hist[d + 3]++;
    }
    printf("  weights: %zu of %u not on the 1/256 grid; on the grid, equal to rounded fraction %zu, truncated %zu; "
           "q - trunc histogram [-3..4]: %d %d %d %d %d %d %d %d\n", not8, n, rnd, trn, hist[0], hist[1], hist[2], hist[3],
           hist[4], hist[5], hist[6], hist[7]);
    c.dev.destroy(sn);
    c.dev.destroy(su);
    c.dev.destroy(im);
    for (Buffer* b : {&bimg, &buv, &bm, &bn, &bq, &braw}) c.dev.destroy(*b);
}

int run_probes(Ctx& c, const fs::path& cases) {
    printf("== probe sampler\n");
    probe_weights(c);
    {
        Lcg r;
        std::vector<uint16_t> img(1024 * 512 * 4);
        for (auto& x : img) x = half_bits_rand(r, true);
        probe_tex(c, img, 1024, 512, "random signed texels 1024x512");
    }
    CaseFile pf;
    if (pf.load(cases / "post_f10")) probe_tex(c, pf.read<uint16_t>("hist.f16"), 2880, 1620, "captured history 2880x1620");
    return run_probes(c);
}

// lane/element layout of the 16x16 fp16 coopmats: the transposed fast path needs an accumulator copied element by
// element to be the B operand of the same matrix (and the A operand of its transpose)
static int probe_wmma(Ctx& c) {
    if (!c.k.has("probe_wmma")) return 0;
    std::vector<uint16_t> m(256);
    for (int i = 0; i < 256; ++i) m[i] = half_rn(i);
    Buffer bm = c.dev.create(512), bb = c.dev.create(512), ba = c.dev.create(512), lh = c.dev.create(1024), la = c.dev.create(1024),
           lb = c.dev.create(1024);
    c.dev.upload(bm, m.data(), 512);
    struct { uint64_t m, b, a, lh, la, lb; } pc{bm.addr, bb.addr, ba.addr, lh.addr, la.addr, lb.addr};
    c.dev.begin();
    c.dev.dispatch(c.k["probe_wmma"], &pc, sizeof pc, 1);
    c.dev.submit();
    std::vector<uint16_t> vb(256), va(256), vh(512), vla(512), vlb(512);
    c.dev.download(bb, vb.data(), 512), c.dev.download(ba, va.data(), 512);
    c.dev.download(lh, vh.data(), 1024), c.dev.download(la, vla.data(), 1024), c.dev.download(lb, vlb.data(), 1024);
    int eb = 0, ea = 0;
    for (int r = 0; r < 16; ++r)
        for (int k = 0; k < 16; ++k) eb += vb[r * 16 + k] != m[r * 16 + k], ea += va[r * 16 + k] != m[k * 16 + r];
    printf("  accumulator copied to B: %d of 256 differ from M; to A: %d of 256 differ from M^T\n", eb, ea);
    for (int l : {0, 1, 15, 16, 31}) {
        printf("  lane %2d (row, col) acc:", l);
        for (int i = 0; i < 8; ++i) { int v = int(to_float(vh[l * 16 + i])); printf(" %d,%d", v / 16, v % 16); }
        printf(" | A:");
        for (int i = 0; i < 8; ++i) { int v = int(to_float(vla[l * 16 + i])); printf(" %d,%d", v / 16, v % 16); }
        printf(" | B:");
        for (int i = 0; i < 8; ++i) { int v = int(to_float(vlb[l * 16 + i])); printf(" %d,%d", v / 16, v % 16); }
        printf("\n");
    }
    for (Buffer* b : {&bm, &bb, &ba, &lh, &la, &lb}) c.dev.destroy(*b);
    return eb != 0 || ea != 0;
}

// WMMA throughput: 8 independent chains per subgroup, timed (TFLOPS at the clock the card runs)
static void probe_rate(Ctx& c) {
    std::vector<uint16_t> m(512, 0x3c00);
    Buffer src = c.dev.create(1024), dst = c.dev.create(size_t(8192) * 1024 * 2);
    c.dev.upload(src, m.data(), 1024);
    const uint32_t groups = 8192, n = 512;
    for (auto [name, k] : {std::pair<const char*, int>{"probe_rate_f16", 16}, {"probe_rate_f32", 16}, {"probe_rate_fp8", 16}, {"probe_rate_fp8k32", 32}}) {
        if (!c.k.has(name)) { printf("  %s: not built\n", name); continue; }
        struct { uint64_t s, d; uint32_t n; } pc{src.addr, dst.addr, n};
        std::vector<double> t;
        for (int r = 0; r < 30; ++r) {
            c.dev.begin();
            c.dev.dispatch(c.k[name], &pc, sizeof pc, groups);
            t.push_back(c.dev.submit()[0]);
        }
        std::sort(t.begin(), t.end());
        double ms = t[t.size() / 2], mm = double(groups) * 4 * 8 * n;
        printf("  %-18s %.3f ms, %.1f TFLOPS, %.2f ns per WMMA per SIMD (128 SIMDs)\n", name, ms, mm * 2 * 16 * 16 * k / (ms * 1e9),
               ms * 1e6 / (mm / 128));
    }
    c.dev.destroy(src), c.dev.destroy(dst);
}

int run_probes(Ctx& c) {
    int bad = 0;
    printf("== probe wmma rate\n");
    probe_rate(c);
    printf("== probe div255\n");
    bad += probe_div(c);
    printf("== probe wmma layout\n");
    bad += probe_wmma(c);
    return bad;
}
