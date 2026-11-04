// SPDX-License-Identifier: MIT
#include <cstdio>
#include <cstring>

#include "harness.hpp"

using namespace onr;

namespace {

template <class T> Buffer make(Ctx& c, const std::vector<T>& v) {
    Buffer b = c.dev.create(v.size() * sizeof(T));
    c.dev.upload(b, v.data(), v.size() * sizeof(T));
    return b;
}

template <class T> std::vector<T> fetch(Ctx& c, const Buffer& b, size_t n) {
    std::vector<T> v(n);
    c.dev.download(b, v.data(), n * sizeof(T));
    return v;
}

void bit_line(const char* what, uint32_t ours, uint32_t ref, const char* fmt = "0x%08x") {
    printf("  %-28s ours ", what);
    printf(fmt, ours);
    printf("  ref ");
    printf(fmt, ref);
    printf("  %s\n", ours == ref ? "equal" : "DIFF");
}

}  // namespace

int run_clear(Ctx& c, const CaseFile& cf) {
    int bad = 0;
    for (auto& r : cf.all("range")) {
        uint32_t count = uint32_t(std::stoul(r[1])), size = uint32_t(std::stoul(r[2]));
        for (uint32_t n : {count, count - 1}) {  // the captured count, plus a ragged one for the tail mask
            Buffer b = c.dev.create(size + 64);
            c.dev.fill(b, 0xa5a5a5a5u);
            c.dev.begin();
            c.k.clear_range(c.dev, {b.addr, n});
            c.dev.submit();
            auto v = fetch<uint8_t>(c, b, size + 64);
            size_t wrong = 0;
            for (uint32_t i = 0; i < size + 64; ++i) wrong += v[i] != (i < n ? 0 : 0xa5);
            printf("  launch %s: %u bytes cleared, %zu wrong bytes (buffer + 64 guard)\n", r[0].c_str(), n, wrong);
            bad += wrong != 0;
            c.dev.destroy(b);
        }
    }
    return bad != 0;
}

int run_pooled(Ctx& c, const CaseFile& cf) {
    I2 in{cf.i("in", 0), cf.i("in", 1)}, out{cf.i("out", 0), cf.i("out", 1)};
    Buffer src = make(c, cf.read<uint32_t>("in.u32"));
    Buffer dst = c.dev.create(size_t(out.x) * out.y * 4);
    PooledHistory2PC p{src.addr, dst.addr, in, out};
    double ms = time_it(c, [&] { c.k.pooled_history2(c.dev, p); });
    auto o = fetch<uint32_t>(c, dst, size_t(out.x) * out.y);
    double f = cmp_1010102("DownsampledHistory2 vs capture", o, cf.read<uint32_t>("exp_out.u32"));
    cmp_1010102("DownsampledHistory2 vs ref-impl", o, cf.read<uint32_t>("ref_out.u32"));
    printf("  gpu %.4f ms\n", ms);
    c.dev.destroy(src);
    c.dev.destroy(dst);
    return f != 1.0;
}

int run_ae(Ctx& c, const CaseFile& cf) {
    I2 size{cf.i("size", 0), cf.i("size", 1)}, grid{cf.i("grid", 0), cf.i("grid", 1)};
    Buffer color = make(c, cf.read<uint16_t>("color.f16"));
    Buffer acc = c.dev.create(16), next = c.dev.create(1028), exp = c.dev.create(4), prev = c.dev.create(4),
           blk = c.dev.create(16);
    c.dev.fill(acc, 0);
    c.dev.fill(next, 0xdeadbeefu);
    uint32_t pv = uint32_t(cf.u("prev"));
    c.dev.upload(prev, &pv, 4);
    AeSumPC s{color.addr, acc.addr, size, {cf.i("origin", 0), cf.i("origin", 1)}, {cf.i("max", 0), cf.i("max", 1)}};
    AeFinalPC f{acc.addr, next.addr, exp.addr, prev.addr, blk.addr, cf.f("inv"), uint32_t(cf.i("reset")), 1028 / 4};
    c.dev.begin();
    c.k.auto_exposure(c.dev, s, f, grid);
    c.dev.submit();
    auto a = fetch<uint64_t>(c, acc, 2);
    auto e = fetch<uint16_t>(c, exp, 1);
    auto b = fetch<uint32_t>(c, blk, 4);
    auto nx = fetch<uint32_t>(c, next, 1028 / 4);
    bool ok = true;
    printf("  sum ours %llu capture %llu ref-impl %llu  blocks %llu/%d  %s\n", (unsigned long long)a[0],
           (unsigned long long)cf.u("exp_total"), (unsigned long long)cf.u("ref_total"), (unsigned long long)a[1],
           cf.i("ncta"), a[0] == cf.u("exp_total") ? "equal" : "DIFF");
    ok &= a[0] == cf.u("exp_total");
    bit_line("exposure fp16 vs capture", e[0], uint32_t(cf.u("exp_exposure")), "0x%04x");
    ok &= e[0] == cf.u("exp_exposure");
    const char* names[4] = {"block[0] exp", "block[1] exp_prev", "block[2] 1/exp", "block[3] log2 ratio/6"};
    for (int i = 0; i < 4; ++i) {
        uint32_t ref = uint32_t(std::stoul(cf.get("exp_block")[i], nullptr, 16));
        bit_line(names[i], b[i], ref);
        if (b[i] != ref) printf("      (%.9g vs %.9g, %d ulp; ref-impl %s)\n", bitsf(b[i]), bitsf(ref), int(b[i] - ref),
                                cf.get("ref_block")[i].c_str());
        ok &= b[i] == ref || i == 3;  // the log2 term differs on the reference GPU's MUFU.LG2 too
    }
    bool cleared = std::all_of(nx.begin(), nx.end(), [](uint32_t x) { return x == 0; });
    printf("  next accumulator cleared: %s\n", cleared ? "yes" : "NO");
    double ms = time_it(c, [&] { c.k.auto_exposure(c.dev, s, f, grid); });
    printf("  gpu %.4f ms (sum + final)\n", ms);
    for (Buffer* x : {&color, &acc, &next, &exp, &prev, &blk}) c.dev.destroy(*x);
    return !(ok && cleared);
}

int run_ds(Ctx& c, const CaseFile& cf) {
    I2 in{cf.i("in_size", 0), cf.i("in_size", 1)}, out{cf.i("out_size", 0), cf.i("out_size", 1)};
    float e = cf.f("exposure");
    Buffer eb = make(c, std::vector<float>{e, 0, 0, 0});
    Buffer dst = c.dev.create(size_t(out.x) * out.y * 8);
    std::vector<Buffer> bufs;
    std::vector<DownsamplePC> ps;
    for (auto& s : cf.all("slice")) {
        auto v = [&](int i) { return std::stoi(s[i]); };
        Buffer rg = make(c, cf.read<uint16_t>("rg" + s[0] + ".f16"));
        Buffer b = make(c, cf.read<uint16_t>("b" + s[0] + ".f16"));
        ps.push_back({rg.addr, b.addr, dst.addr, eb.addr, in, out, {v(1), v(2)}, {v(3), v(4)}, {v(5), v(6)}, {v(7), v(8)}});
        bufs.push_back(rg);
        bufs.push_back(b);
    }
    double ms = time_it(c, [&] { for (auto& p : ps) c.k.downsample(c.dev, p); });
    auto o = fetch<uint16_t>(c, dst, size_t(out.x) * out.y * 4);
    double f = cmp_f16("output rgba16f vs capture", o, cf.read<uint16_t>("exp_out.f16"));
    cmp_f16("output rgba16f vs ref-impl", o, cf.read<uint16_t>("ref_out.f16"));
    printf("  gpu %.4f ms (4 slices)\n", ms);
    for (auto& b : bufs) c.dev.destroy(b);
    c.dev.destroy(eb);
    c.dev.destroy(dst);
    return f != 1.0;
}
