// SPDX-License-Identifier: MIT
// Transformer block (FP16 WMMA) of one level: TensorOutput against the capture and ref-impl, in dB.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "harness.hpp"
#include "opennr_rr/engine.hpp"

using namespace onr;

static float e4m3_sm86(uint8_t b) {
    int e = (b >> 3) & 15, m = b & 7;
    float v = std::ldexp(1.0f + float(m) / 8.0f, (e ? e : 0) - 7);
    return b & 0x80 ? -v : v;
}

double cmp_e4m3(const char* what, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    size_t n = std::min(a.size(), b.size()), eq = 0, one = 0;
    double mse = 0, peak = 0;
    for (size_t i = 0; i < n; ++i) {
        eq += a[i] == b[i];
        one += a[i] != b[i] && (a[i] & 0x80) == (b[i] & 0x80) && std::abs(int(a[i] & 0x7f) - int(b[i] & 0x7f)) == 1;
        double x = e4m3_sm86(a[i]), y = e4m3_sm86(b[i]);
        mse += (x - y) * (x - y);
        peak = std::max(peak, std::fabs(y));
    }
    mse /= double(n);
    double psnr = mse == 0 ? INFINITY : 10 * std::log10(peak * peak / mse);
    printf("  %-40s e4m3 equal %.4f%%  1-ulp %.4f%%  psnr %.2f dB\n", what, 100.0 * double(eq) / double(n),
           100.0 * double(one) / double(n), psnr);
    return psnr;
}

// canonical (arena channel order) weights -> the logical order the block kernels use: K of the matrices that
// read the residual stream and N of those that write it go to the GPU fragment order (FASE3B section 4)
static int arena_channel(int c) {
    int g = c & 31;
    return (c & ~31) + ((g >> 3) & 1) * 16 + ((g >> 4) & 1) * 2 + 4 * ((g & 7) >> 1) + (g & 1);
}

static std::vector<uint16_t> perm_k(const std::vector<uint16_t>& w, int K) {
    std::vector<uint16_t> r(w.size());
    for (size_t n = 0; n < w.size() / K; ++n)
        for (int k = 0; k < K; ++k) r[n * K + k] = w[n * K + arena_channel(k)];
    return r;
}

static std::vector<uint16_t> perm_n(const std::vector<uint16_t>& w, int N) {
    size_t K = w.size() / N;
    std::vector<uint16_t> r(w.size());
    for (int n = 0; n < N; ++n)
        for (size_t k = 0; k < K; ++k) r[n * K + k] = w[size_t(arena_channel(n)) * K + k];
    return r;
}

// block weights of a case (canonical -> logical order), window geometry and the k8 masks; x/y: tensor planes
EncBlockPC load_block(Ctx& c, const CaseFile& cf, std::vector<Buffer>& all, uint64_t x, uint64_t y) {
    I2 size{cf.i("size", 0), cf.i("size", 1)}, shift{cf.i("shift", 0), cf.i("shift", 1)};
    int C = cf.i("dims", 0), X = cf.i("dims", 1);
    auto make = [&](const void* d, size_t n) {
        Buffer b = c.dev.create(n);
        c.dev.upload(b, d, n);
        all.push_back(b);
        return b;
    };
    auto up = [&](const char* f) { auto v = cf.read<uint8_t>(f); return make(v.data(), v.size()); };
    auto upw = [&](std::vector<uint16_t> v) { return make(v.data(), v.size() * 2); };
    auto w = [&](const char* f) { return cf.read<uint16_t>(f); };
    std::vector<uint16_t> masks(512);
    for (int i = 0; i < 256; ++i) { masks[i] = (i % 16) < 8 ? 0x3c00 : 0; masks[256 + i] = (i % 16) < 8 ? 0 : 0x3c00; }
    auto upqv = [&](std::vector<uint16_t> v) { auto d = qv_bytes(v.data(), v.size()); return make(d.data(), d.size()); };
    return EncBlockPC{x, y, upqv(perm_k(w("w_attn_in_a.f16"), C)).addr, upqv(perm_k(w("w_attn_in_b.f16"), C)).addr,
                      upw(with_tiles(perm_n(w("w_attn_out.f16"), C), X)).addr, upw(perm_n(w("w_attn_out_b.f16"), C)).addr,
                      upw(with_tiles(perm_k(w("w_mlp_fc1.f16"), C), C)).addr, up("w_mlp_fc1_b.f16").addr,
                      upw(with_tiles(perm_n(w("w_mlp_fc2.f16"), C), 4 * C)).addr, upw(perm_n(w("w_mlp_fc2_b.f16"), C)).addr,
                      up("w_attn_bias.f16").addr, size, shift,
                      uint32_t(cf.u("ls", 0)) | uint32_t(cf.u("ls", 1)) << 16, (size.x + shift.x + 7) / 8,
                      upw(masks).addr};
}

int run_enc(Ctx& c, const CaseFile& cf) {
    if (cf.kv.count("enc0")) return run_enc0(c, cf);
    if (cf.kv.count("merge_in") || cf.kv.count("dec0") || cf.kv.count("head")) return run_net(c, cf);
    I2 size{cf.i("size", 0), cf.i("size", 1)}, shift{cf.i("shift", 0), cf.i("shift", 1)};
    int C = cf.i("dims", 0), X = cf.i("dims", 1);
    std::string kn = "enc_block_" + std::to_string(C) + "_" + std::to_string(X);
    if (!c.k.has(kn)) {
        printf("  no kernel %s\n", kn.c_str());
        return 1;
    }
    std::vector<Buffer> all;
    auto make = [&](const void* d, size_t n) {
        Buffer b = c.dev.create(n);
        c.dev.upload(b, d, n);
        all.push_back(b);
        return b;
    };
    auto up = [&](const char* f) { auto v = cf.read<uint8_t>(f); return make(v.data(), v.size()); };
    auto upw = [&](std::vector<uint16_t> v) { return make(v.data(), v.size() * 2); };
    Buffer x = up("x.u8");
    Buffer y = c.dev.create(x.size);
    all.push_back(y);
    EncBlockPC p = load_block(c, cf, all, x.addr, y.addr);
    uint32_t windows = uint32_t(p.nwx * ((size.y + shift.y + 7) / 8));
    int bad = 0;
    for (const char* sfx : {"", "_k8", "_k8_x", "_k8_e"}) {
        bool k8 = *sfx != 0;
        if (!c.k.has(kn + sfx)) continue;
        printf("  -- %s\n", !k8 ? "fp32 accumulators" : sfx[3] == 'e' ? "every product as the exact scalar HMMA chain" : sfx[3] ? "k8 + exact scalar P.V chain (HMMA truncation emulated)"
                                                                 : "k8: 8 K per step, accumulator rounded to fp16 (reference HMMA chain)");
        c.dev.fill(y, 0);
        double ms = time_it(c, [&] { c.k.run(c.dev, kn + sfx, &p, sizeof p, windows); });
        auto o = std::vector<uint8_t>(x.size);
        c.dev.download(y, o.data(), o.size());
        if (cf.has_file("exp.u8")) bad += cmp_e4m3("TensorOutput vs capture", o, cf.read<uint8_t>("exp.u8")) < 50.0;
        if (cf.has_file("ref.u8")) {
            double db = cmp_e4m3("TensorOutput vs ref-impl", o, cf.read<uint8_t>("ref.u8"));
            if (!cf.has_file("exp.u8")) bad += db < 50.0;   // no capture tensor for this block: the bar is against ref-impl
        }
        printf("  gpu %.4f ms (%u windows of 64 tokens, C %d, X %d)\n", ms, windows, C, X);
        if (getenv("ONR_ENC_DEBUG")) {
            FILE* f = fopen((cf.dir / (k8 ? "gpu_out_k8.u8" : "gpu_out.u8")).string().c_str(), "wb");
            fwrite(o.data(), 1, o.size(), f);
            fclose(f);
        }
        printf("  %s\n", c.dev.stats(c.k[kn + sfx]).c_str());
    }

    // whole rr2_encN kernel (block k8 + DFN + pooling + merge) against the captured stage outputs
    if (cf.kv.count("stage")) {
        int Co = cf.i("stage", 0);
        I2 hs{size.x / 2, size.y / 2};
        auto dfn_in = cf.read<uint16_t>("dfn_in.f16");
        Buffer dout = c.dev.create(dfn_in.size() * 2), dpool = c.dev.create(size_t(hs.x) * hs.y * 8 + 8);
        Buffer pooled = c.dev.create(size_t(hs.x) * hs.y * std::max(Co, 32) + 8);
        all.push_back(dout); all.push_back(dpool); all.push_back(pooled);
        std::vector<uint16_t> zero(16, 0);
        EncStagePC s{p, up("w_stage_dfn.f16").addr, up("w_stage_dfn_b.f16").addr,
                     Co ? up("w_stage_merge.f16").addr : upw(zero).addr, Co ? up("w_stage_merge_b.f16").addr : upw(zero).addr,
                     upw(dfn_in).addr, dout.addr, dpool.addr, pooled.addr, uint32_t(std::stoul(cf.get("stage").at(1), nullptr, 16))};
        for (Buffer* b : {&y, &dout, &dpool, &pooled}) c.dev.fill(*b, 0);
        std::string sk = "enc_stage_" + std::to_string(C) + "_" + std::to_string(X) + "_" + std::to_string(Co) + (getenv("ONR_EXACT") ? "_e" : getenv("ONR_PVX") ? "_x" : "");
        double ms = time_it(c, [&] { c.k.run(c.dev, sk, &s, sizeof s, windows); });
        printf("  -- whole kernel %s (block k8 + DFN head + DFN pooling%s), %.4f ms\n", sk.c_str(), Co ? " + merge" : "", ms);
        auto o = std::vector<uint8_t>(x.size);
        c.dev.download(y, o.data(), o.size());
        cmp_e4m3("TensorOutput vs capture", o, cf.read<uint8_t>("exp.u8"));
        std::vector<uint16_t> d(dfn_in.size());
        c.dev.download(dout, d.data(), d.size() * 2);
        bad += cmp_f16("DFNOutput rgb vs capture", d, cf.read<uint16_t>("exp_dfn.f16"), 4, 3) < 0.98;   // prototype bar (block not bit-exact)
        if (Co) {
            std::vector<uint8_t> po(size_t(hs.x) * hs.y * Co);
            c.dev.download(pooled, po.data(), po.size());
            bad += cmp_e4m3("PooledOutput vs capture", po, cf.read<uint8_t>("exp_pooled.u8")) < 50.0;
            std::vector<uint16_t> dp(size_t(hs.x) * hs.y * 4);
            c.dev.download(dpool, dp.data(), dp.size() * 2);
            bad += cmp_f16("DFNPooledOutput rgb vs capture", dp, cf.read<uint16_t>("exp_dfn_pooled.f16"), 4, 3) < 0.98;
        }
    }

    // FP8 WMMA experiment for the first two products (see enc_qv_fp8.comp)
    if (c.k.has("enc_qv_fp8") && C == 128 && X == 128) {
        float tab[127];
        for (int b = 0; b < 127; ++b) tab[b] = (b >> 3) ? std::ldexp(1.0f + float(b & 7) / 8.0f, (b >> 3) - 7) : float(b & 7) * std::ldexp(1.0f, -9);
        auto e4 = [&](const char* f) {
            auto h = cf.read<uint16_t>(f);
            std::vector<uint8_t> v(h.size());
            for (size_t i = 0; i < h.size(); ++i) {
                float w = to_float(h[i]), a = std::fabs(w);
                int code = -1;
                for (int b = 0; b < 127 && code < 0; ++b) if (tab[b] == a) code = b;
                if (code < 0) throw std::runtime_error("weight not representable in e4m3");
                v[i] = uint8_t(code | (std::signbit(w) ? 0x80 : 0));
            }
            Buffer b = c.dev.create(v.size());
            c.dev.upload(b, v.data(), v.size());
            all.push_back(b);
            return b;
        };
        Buffer qv = c.dev.create(size_t(windows) * 64 * 2 * X * 2);
        all.push_back(qv);
        EncQvFp8PC q{x.addr, qv.addr, e4("w_attn_in_a.f16").addr, e4("w_attn_in_b.f16").addr, size, shift, p.nwx};
        double ms8 = time_it(c, [&] { c.k.enc_qv_fp8(c.dev, q, windows); });
        std::vector<uint8_t> d(qv.size);
        c.dev.download(qv, d.data(), d.size());
        FILE* f = fopen((cf.dir / "gpu_qv8.f16").string().c_str(), "wb");
        fwrite(d.data(), 1, d.size(), f);
        fclose(f);
        printf("  fp8 Q/V products: %.4f ms, written to gpu_qv8.f16 (compare: gpu/tools/enc_model.py <case> qv8)\n", ms8);
    }
    for (auto& b : all) c.dev.destroy(b);
    if (cf.kv.count("dec1")) bad += run_net(c, cf);
    return bad;
}
