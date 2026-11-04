// SPDX-License-Identifier: MIT
// Network kernels besides the enc2..enc5 stages: enc1 (input merge), decN_0 (up + skip), decN_1 (decoder DFN),
// the dec0_1 head; and a whole frame on the GPU from the captured enc0 outputs (run_chain), every kernel fed
// by our own previous outputs, compared stage by stage with the capture and at the end with the final output.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "d3d12_amd.hpp"
#include "harness.hpp"
#include "opennr_rr/engine.hpp"

using namespace onr;

namespace {

struct Pool {
    Ctx& c;
    std::vector<Buffer> v;
    explicit Pool(Ctx& c_) : c(c_) {}
    ~Pool() { for (auto& b : v) c.dev.destroy(b); }
    Buffer zero(size_t n) {
        Buffer b = c.dev.create(n + 16);
        c.dev.fill(b, 0);
        v.push_back(b);
        return b;
    }
    template <class T> Buffer vec(const std::vector<T>& d) {
        Buffer b = c.dev.create(d.size() * sizeof(T));
        c.dev.upload(b, d.data(), d.size() * sizeof(T));
        v.push_back(b);
        return b;
    }
    Buffer file(const CaseFile& cf, const std::string& f) { return vec(cf.read<uint8_t>(f)); }
};

struct Stage {
    std::string name, kernel;
    std::vector<uint8_t> pc;
    uint32_t groups = 0;
    template <class P> void set(const P& p) {
        pc.resize(sizeof p);
        memcpy(pc.data(), &p, sizeof p);
    }
    void record(Ctx& c) const { c.k.run(c.dev, kernel, pc.data(), uint32_t(pc.size()), groups); }
};

struct Out {
    Buffer tensor, pooled, dfn, dfn_pool, ajab, mvf, hidden, aniso;
    I2 size{};
    int C = 0, Co = 0, out_h = 0;
};

uint32_t windows(const EncBlockPC& b) { return uint32_t(b.nwx * ((b.size.y + b.shift.y + 7) / 8)); }
uint32_t hex(const CaseFile& cf, const char* k, size_t n) { return uint32_t(std::stoul(cf.get(k).at(n), nullptr, 16)); }
I2 size_of(const CaseFile& cf) { return {cf.i("size", 0), cf.i("size", 1)}; }
std::string dims(const CaseFile& cf) { return std::to_string(cf.i("dims", 0)) + "_" + std::to_string(cf.i("dims", 1)); }
// ONR_PVX=1: the kernels with the exact scalar P.V chain (suffix _x)
// ONR_EXACT=1: every product as the exact scalar HMMA chain (suffix _e)
// ONR_F32=1: the fast variant, fp32 accumulation over each product's whole K (suffix _f)
// post_f reads InternalHistory through the sampler as the runtime does (ONR_POST_HW=0: by hand)
bool post_hw(Ctx&) { const char* e = getenv("ONR_POST_HW"); return getenv("ONR_F32") && !(e && !strcmp(e, "0")); }

std::string sfx() { return getenv("ONR_EXACT") ? "_e" : getenv("ONR_PVX") ? "_x" : getenv("ONR_F32") ? "_f" : ""; }

// the runtime-path dec0_1 kernels (k8, _f) write the post tile themselves; returns whether post_tile is skipped
bool fuse_tile(std::vector<Stage>& st, uint64_t tile) {
    std::string s = sfx();
    if (s != "" && s != "_f") return false;
    for (auto& x : st)
        if (x.name == "dec0_1") {
            memcpy(x.pc.data() + offsetof(DecDfnPC, tile), &tile, 8);
            return true;
        }
    return false;
}

template <class T> std::vector<T> fetch(Ctx& c, const Buffer& b, size_t n) {
    std::vector<T> v(n);
    c.dev.download(b, v.data(), n * sizeof(T));
    return v;
}

double cmp_u8(const char* what, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, bool sgn) {
    size_t n = std::min(a.size(), b.size()), eq = 0;
    int maxd = 0;
    for (size_t i = 0; i < n; ++i) {
        eq += a[i] == b[i];
        int x = sgn ? int(int8_t(a[i])) : a[i], y = sgn ? int(int8_t(b[i])) : b[i];
        maxd = std::max(maxd, std::abs(x - y));
    }
    double f = double(eq) / double(n ? n : 1);
    printf("  %-44s codes equal %.6f%% (%zu of %zu differ, max %d)\n", what, 100 * f, n - eq, n, maxd);
    return f;
}

// rr2_enc1..5: x = input planes (enc1: the enc0 TensorOutput), dfn_in = DFN image of the level
Stage enc_stage(Pool& P, const CaseFile& cf, uint64_t x, uint64_t dfn_in, Out& o) {
    I2 size = size_of(cf), hs{size.x / 2, size.y / 2};
    int C = cf.i("dims", 0), Co = cf.i("stage", 0);
    o.size = size, o.C = C, o.Co = Co;
    o.tensor = P.zero(size_t(size.x) * size.y * C);
    o.dfn = P.zero(size_t(size.x) * size.y * 8);
    o.pooled = P.zero(size_t(hs.x) * hs.y * std::max(Co, 32));
    o.dfn_pool = P.zero(size_t(hs.x) * hs.y * 8);
    auto mw = Co ? cf.read<uint16_t>("w_stage_merge.f16") : std::vector<uint16_t>(16, 0);
    auto mb = Co ? cf.read<uint16_t>("w_stage_merge_b.f16") : std::vector<uint16_t>(16, 0);
    I2 in{};
    if (cf.kv.count("merge_in")) {
        auto a = cf.read<uint16_t>("w_merge_in.f16"), b = cf.read<uint16_t>("w_merge_in_b.f16");
        mw.insert(mw.end(), a.begin(), a.end());
        mb.insert(mb.end(), b.begin(), b.end());
        in = {cf.i("merge_in", 0), cf.i("merge_in", 1)};
    }
    if (Co && !cf.kv.count("merge_in")) mw = with_tiles(mw, 4 * size_t(cf.i("dims", 0)));
    EncStagePC s{load_block(P.c, cf, P.v, x, o.tensor.addr), P.file(cf, "w_stage_dfn.f16").addr, P.file(cf, "w_stage_dfn_b.f16").addr,
                 P.vec(mw).addr, P.vec(mb).addr, dfn_in, o.dfn.addr, o.dfn_pool.addr, o.pooled.addr, hex(cf, "stage", 1), in};
    Stage st;
    st.kernel = "enc_stage_" + dims(cf) + "_" + std::to_string(Co) + sfx();
    st.set(s);
    st.groups = windows(s.b);
    return st;
}

// rr2_decN_kernel_0: skip = encoder TensorOutput of the level, xlo = tensor of the level below
Stage dec_up(Pool& P, const CaseFile& cf, uint64_t skip, uint64_t xlo, Out& o) {
    I2 size = size_of(cf);
    int C = cf.i("dims", 0), Cp = cf.i("dec0", 0);
    o.size = size, o.C = C;
    o.tensor = P.zero(size_t(size.x) * size.y * C);
    DecUpPC p{load_block(P.c, cf, P.v, skip, o.tensor.addr), xlo, P.file(cf, "w_up.f16").addr, P.file(cf, "w_up_b.f16").addr,
              {size.x / 2, size.y / 2}, hex(cf, "dec0", 1)};
    Stage st;
    st.kernel = "dec_up_" + dims(cf) + "_" + std::to_string(Cp) + sfx();
    st.set(p);
    st.groups = windows(p.b);
    return st;
}

// rr2_decN_kernel_1 (DFN head) or rr2_dec0_kernel_1 (output head): x = sub0 tensor, below = DFN image of the level
// below, skip = encoder DFN image of the level
Stage dec_dfn(Pool& P, const CaseFile& cf, uint64_t x, uint64_t below, uint64_t skip, Out& o) {
    bool head = cf.kv.count("head") != 0;
    I2 size = size_of(cf);
    int C = cf.i("dims", 0);
    o.size = size, o.C = C;
    o.out_h = head ? cf.i("head", 3) : cf.i("dec1", 1);
    size_t px = size_t(size.x) * o.out_h;
    if (!head) o.tensor = P.zero(size_t(size.x) * size.y * C);
    o.dfn = P.zero(px * 8);
    DecDfnPC p{};
    p.b = load_block(P.c, cf, P.v, x, head ? 0 : o.tensor.addr);
    if (!head) {
        p.dfn_w = P.file(cf, "w_dfn.f16").addr;
        p.dfn_b = P.file(cf, "w_dfn_b.f16").addr;
    }
    p.below = below, p.skip = skip, p.dfn_out = o.dfn.addr;
    p.lo_size = {size.x / 2, size.y / 2};
    p.s_dfn = hex(cf, head ? "head" : "dec1", 0);
    p.out_h = o.out_h;
    if (head) {
        o.ajab = P.zero(px * 4), o.mvf = P.zero(px * 4), o.hidden = P.zero(px * 4), o.aniso = P.zero(px * 8);
        p.ajab = o.ajab.addr, p.mvf = o.mvf.addr, p.hidden = o.hidden.addr, p.aniso = o.aniso.addr;
        p.k_scale = cf.f("head", 1), p.k_tanh = cf.f("head", 2), p.out_w = size.x;
    }
    Stage st;
    st.kernel = (head ? "dec_head_" + dims(cf) : "dec_dfn_" + dims(cf)) + sfx();
    st.set(p);
    st.groups = windows(p.b);
    return st;
}

// compares whatever the kernel produced with the captured outputs of its case; returns the count below the bars
int check(Ctx& c, const CaseFile& cf, const Out& o, const std::string& tag) {
    int bad = 0;
    auto line = [&](const char* w) { return tag + " " + w; };
    size_t n = size_t(o.size.x) * o.size.y;
    if (o.tensor.buf && cf.has_file("exp.u8"))
        bad += cmp_e4m3(line("TensorOutput").c_str(), fetch<uint8_t>(c, o.tensor, n * o.C), cf.read<uint8_t>("exp.u8")) < 50.0;
    if (o.Co)
        bad += cmp_e4m3(line("PooledOutput").c_str(), fetch<uint8_t>(c, o.pooled, n / 4 * o.Co), cf.read<uint8_t>("exp_pooled.u8")) < 50.0;
    if (o.dfn.buf) {
        bool enc = cf.kv.count("stage") != 0;   // encoder DFN: alpha is image.a * 0, compare rgb as the earlier rounds
        size_t px = o.out_h ? size_t(o.size.x) * o.out_h : n;
        bad += cmp_f16(line(enc ? "DFNOutput rgb" : "DFNOutput rgba").c_str(), fetch<uint16_t>(c, o.dfn, px * 4),
                       cf.read<uint16_t>("exp_dfn.f16"), 4, enc ? 3 : 4) < 0.98;
        if (o.Co)
            bad += cmp_f16(line("DFNPooledOutput rgb").c_str(), fetch<uint16_t>(c, o.dfn_pool, n), cf.read<uint16_t>("exp_dfn_pooled.f16"), 4, 3) < 0.98;
    }
    if (o.ajab.buf) {
        size_t px = size_t(o.size.x) * o.out_h;
        auto an = fetch<uint16_t>(c, o.aniso, px * 4), ea = cf.read<uint16_t>("exp_aniso.f16");
        bad += cmp_f16(line("Aniso rgba").c_str(), an, ea) < 0.99;
        cmp_f16(line("Aniso z (a b tanh)").c_str(), std::vector<uint16_t>(an.begin() + 2, an.end()), std::vector<uint16_t>(ea.begin() + 2, ea.end()), 4, 1);
        bad += cmp_u8(line("AjabGateOgate").c_str(), fetch<uint8_t>(c, o.ajab, px * 4), cf.read<uint8_t>("exp_ajab.u8"), false) < 0.99;
        bad += cmp_u8(line("MvecFactors").c_str(), fetch<uint8_t>(c, o.mvf, px * 4), cf.read<uint8_t>("exp_mvf.u8"), false) < 0.99;
        bad += cmp_u8(line("HiddenHistory").c_str(), fetch<uint8_t>(c, o.hidden, px * 4), cf.read<uint8_t>("exp_hidden.u8"), true) < 0.99;
    }
    return bad;
}

// rr2_enc0 inputs (device addresses) and the per-pixel surfaces it writes
struct Enc0In {
    uint64_t color, albedo, spec, normals, rough, depth, mv, hit, pm1, hidden, ds, ds2, expo, prm;
    I2 ds_size, ds2_size;
};
struct Enc0Out {
    Out net;   // tensor (1280 x 736 x 32), dfn (1280 x 736), dfn_pool (640 x 368)
    Buffer attr, img, pmv2, ero, nond;
};

// both halves of rr2_enc0: enc0_attr (per pixel) and enc0_stage_32_64 (dither, embed, block, DFN)
std::vector<Stage> enc0_stages(Pool& P, const CaseFile& cf, const Enc0In& in, Enc0Out& o) {
    I2 size = size_of(cf), img{size.x, 720};
    size_t px = size_t(img.x) * img.y;
    o.attr = P.zero(px * 48 * 2), o.img = P.zero(size_t(size.x) * size.y * 8);
    o.pmv2 = P.zero(px * 8), o.ero = P.zero(px * 4), o.nond = P.zero(px * 4);
    Out& n = o.net;
    n.size = size, n.C = 32, n.Co = 0;
    n.tensor = P.zero(size_t(size.x) * size.y * 32);
    n.dfn = P.zero(size_t(size.x) * size.y * 8);
    n.dfn_pool = P.zero(size_t(size.x / 2) * (size.y / 2) * 8);
    Enc0AttrPC a{in.color, in.albedo, in.spec, in.normals, in.rough, in.depth, in.mv, in.hit, in.pm1, in.hidden, in.ds, in.ds2,
                 in.expo, in.prm, P.file(cf, "tab.f32").addr, o.attr.addr, o.pmv2.addr, o.ero.addr, o.nond.addr, o.img.addr,
                 img, in.ds_size, in.ds2_size, uint32_t(cf.i("enc0", 1)), size, I2{2880, 1620}, 0, 0, P.file(cf, "lut.u16").addr,
                 uint32_t(cf.i("enc0", 0))};
    Enc0StagePC e{};
    e.s = EncStagePC{load_block(P.c, cf, P.v, 0, n.tensor.addr), P.file(cf, "w_stage_dfn.f16").addr, P.file(cf, "w_stage_dfn_b.f16").addr,
                     P.file(cf, "w_embed.f16").addr, P.file(cf, "w_embed_b.f16").addr, o.img.addr, n.dfn.addr, n.dfn_pool.addr, 0,
                     hex(cf, "stage", 1), I2{}};
    e.attr = o.attr.addr, e.lut = P.file(cf, "lut.u16").addr, e.frame8 = uint32_t(cf.i("enc0", 0)), e.img_h = img.y, e.img_w = img.x;
    std::vector<Stage> st(2);
    st[0].name = "enc0_attr", st[0].kernel = sfx() == "_f" ? "enc0_attr_f" : "enc0_attr", st[0].groups = uint32_t((size.x * size.y + 63) / 64);
    st[0].set(a);
    st[1].name = "enc0", st[1].kernel = "enc0_stage_32_64" + sfx(), st[1].groups = windows(e.s.b);
    st[1].set(e);
    return st;
}

Enc0In enc0_inputs(Pool& P, const CaseFile& cf) {
    Enc0In in{};
    in.color = P.file(cf, "color.f16").addr, in.albedo = P.file(cf, "albedo.u8").addr, in.spec = P.file(cf, "spec.u8").addr;
    in.normals = P.file(cf, "normals.f16").addr, in.rough = P.file(cf, "rough.f16").addr, in.depth = P.file(cf, "depth.f32").addr;
    in.mv = P.file(cf, "mv.f16").addr, in.hit = P.file(cf, "hit.f16").addr, in.prm = P.file(cf, "prm.f32").addr;
    in.ds_size = {cf.i("ds", 0), cf.i("ds", 1)}, in.ds2_size = {cf.i("ds", 2), cf.i("ds", 3)};
    return in;
}

int check_enc0(Ctx& c, const CaseFile& cf, const Enc0Out& o, const std::string& tag) {
    int bad = 0;
    size_t px = 1280 * 720;
    I2 s = o.net.size;
    if (cf.has_file("ref_attr.f16") && sfx() != "_f") {   // the fast record is dithered (ATTR64)
        auto g = fetch<uint16_t>(c, o.attr, px * 48), r = cf.read<uint16_t>("ref_attr.f16");
        std::vector<uint16_t> a(px * 43);
        for (size_t i = 0; i < px; ++i) memcpy(&a[i * 43], &g[i * 48], 43 * 2);
        cmp_f16((tag + " attributes vs NumPy (43, before dither)").c_str(), a, r);
        std::vector<size_t> col(43);
        for (size_t i = 0; i < a.size(); ++i) col[i % 43] += a[i] != r[i];
        printf("    differing values per attribute:");
        for (int k = 0; k < 43; ++k) if (col[k]) printf(" %d:%zu", k, col[k]);
        printf("\n");
    }
    bad += cmp_e4m3((tag + " TensorOutput").c_str(), fetch<uint8_t>(c, o.net.tensor, size_t(s.x) * s.y * 32), cf.read<uint8_t>("exp.u8")) < 45.0;
    bad += cmp_f16((tag + " DFNOutput rgb").c_str(), fetch<uint16_t>(c, o.net.dfn, size_t(s.x) * s.y * 4), cf.read<uint16_t>("exp_dfn.f16"), 4, 3) < 0.95;
    bad += cmp_f16((tag + " DFNPooledOutput rgb").c_str(), fetch<uint16_t>(c, o.net.dfn_pool, size_t(s.x) * s.y), cf.read<uint16_t>("exp_dfn_pooled.f16"), 4, 3) < 0.95;
    if (cf.has_file("exp_pmv2.f16")) {
        cmp_f16((tag + " PrevMvec2").c_str(), fetch<uint16_t>(c, o.pmv2, px * 4), cf.read<uint16_t>("exp_pmv2.f16"), 4);
        cmp_f16((tag + " PrevMvecEroded").c_str(), fetch<uint16_t>(c, o.ero, px * 2), cf.read<uint16_t>("exp_ero.f16"), 2);
        cmp_f16((tag + " PrevMvecNondilated").c_str(), fetch<uint16_t>(c, o.nond, px * 2), cf.read<uint16_t>("exp_nond.f16"), 2);
    }
    return bad;
}

}  // namespace

int run_enc0(Ctx& c, const CaseFile& cf) {
    Pool P(c);
    Enc0In in = enc0_inputs(P, cf);
    in.pm1 = P.file(cf, "pm1.f16").addr, in.hidden = P.file(cf, "hidden.u8").addr, in.ds = P.file(cf, "ds.u32").addr;
    in.ds2 = P.file(cf, "ds2.u32").addr, in.expo = P.file(cf, "expo.f32").addr;
    Enc0Out o;
    auto st = enc0_stages(P, cf, in, o);
    for (auto& s : st)
        if (!c.k.has(s.kernel)) { printf("  no kernel %s\n", s.kernel.c_str()); return 1; }
    double ms = time_it(c, [&] { st[0].record(c); c.dev.barrier(); st[1].record(c); });
    printf("  -- rr2_enc0: %s + %s (%u windows): %.4f ms\n", st[0].kernel.c_str(), st[1].kernel.c_str(), st[1].groups, ms);
    int bad = check_enc0(c, cf, o, "");
    if (const char* d = getenv("ONR_ENC0_DUMP")) {   // raw attributes + PrevMvec2, to compare two builds bit for bit
        auto g = fetch<uint16_t>(c, o.attr, size_t(1280) * 720 * 48), m = fetch<uint16_t>(c, o.pmv2, size_t(1280) * 720 * 4);
        g.insert(g.end(), m.begin(), m.end());
        FILE* f = fopen(d, "wb");
        fwrite(g.data(), 2, g.size(), f);
        fclose(f);
    }
    printf("  %s\n  %s\n", c.dev.stats(c.k[st[0].kernel]).c_str(), c.dev.stats(c.k[st[1].kernel]).c_str());
    return bad;
}

namespace {

// the kernel of an enc1 / decN_0 / decN_1 / dec0_1 case with its inputs in P
Stage net_stage(Pool& P, const CaseFile& cf, Out& o, const char*& what) {
    if (cf.kv.count("merge_in")) {
        what = "input merge + block k8 + DFN + pooling + merge";
        return enc_stage(P, cf, P.file(cf, "x.u8").addr, P.file(cf, "dfn_in.f16").addr, o);
    }
    if (cf.kv.count("dec0")) {
        what = "up + skip + block k8";
        return dec_up(P, cf, P.file(cf, "x.u8").addr, P.file(cf, "xlo.u8").addr, o);
    }
    what = cf.kv.count("head") ? "block k8 + output head" : "block k8 + decoder DFN";
    return dec_dfn(P, cf, P.file(cf, "x.u8").addr, P.file(cf, "below.f16").addr, P.file(cf, "skip.f16").addr, o);
}

}  // namespace

int run_net(Ctx& c, const CaseFile& cf) {
    Pool P(c);
    Out o;
    const char* what;
    Stage st = net_stage(P, cf, o, what);
    if (!c.k.has(st.kernel)) {
        printf("  no kernel %s\n", st.kernel.c_str());
        return 1;
    }
    double ms = time_it(c, [&] { st.record(c); });
    printf("  -- whole kernel %s (%s, %u windows): %.4f ms\n", st.kernel.c_str(), what, st.groups, ms);
    int bad = check(c, cf, o, "");
    printf("  %s\n", c.dev.stats(c.k[st.kernel]).c_str());
    return bad;
}

// FASE9-D3D12 prototype: the same kernel on Vulkan and on a D3D12 device (the Vulkan driver's pipeline ELF handed to
// the D3D12 driver), same inputs, interleaved timing, every buffer compared byte for byte afterwards. Needs ONR_ISA
// (the ELF comes from the driver's internal representations). ONR_WARM recordings first (default 200), ONR_ROUNDS
// alternations (default 3), each recording ONR_DISPATCHES back-to-back dispatches (default 10).
int run_d3d12(Ctx& c, const CaseFile& cf) {
    const char* isa = getenv("ONR_D3D12_ELF") ? getenv("ONR_D3D12_ELF") : getenv("ONR_ISA");   // ONR_D3D12_ELF: ELF dumps from elsewhere
    if (!isa) { printf("  ONR_ISA=<dir> needed\n"); return 1; }
    auto env = [](const char* k, int d) { const char* v = getenv(k); return v ? atoi(v) : d; };
    const int warm = env("ONR_WARM", 200), rounds = env("ONR_ROUNDS", 3), nd = env("ONR_DISPATCHES", 10);
    Pool P(c);
    Out o;
    const char* what;
    Stage st = net_stage(P, cf, o, what);
    if (!c.k.has(st.kernel)) { printf("  no kernel %s\n", st.kernel.c_str()); return 1; }

    std::vector<uint8_t> elf;
    for (int i = 0; i < 4 && elf.empty(); ++i) {
        std::ifstream f(fs::path(isa) / (st.kernel + "." + std::to_string(i) + ".txt"), std::ios::binary);
        std::vector<uint8_t> d((std::istreambuf_iterator<char>(f)), {});
        if (d.size() > 6 && !memcmp(d.data(), "ELF.cs", 6)) {
            const uint8_t m[] = {0x7f, 'E', 'L', 'F'};
            elf.assign(std::search(d.begin(), d.end(), m, m + 4), d.end());
        }
    }
    if (elf.empty()) { printf("  no ELF dump of %s in %s\n", st.kernel.c_str(), isa); return 1; }

    onr::d3d::Device D;
    std::string err;
    if (!D.init(err)) { printf("  d3d12: %s\n", err.c_str()); return 1; }
    std::vector<std::pair<Buffer, onr::d3d::Buffer>> map;   // every buffer of the case, copied before any run
    for (auto& b : P.v) {
        std::vector<uint8_t> h(b.size);
        c.dev.download(b, h.data(), b.size);
        auto db = D.create(b.size);
        D.upload(db, h.data(), b.size);
        map.push_back({b, db});
    }
    std::vector<uint8_t> pc = st.pc;   // device addresses -> the D3D12 copies
    int moved = 0;
    for (size_t off = 0; off + 8 <= pc.size(); off += 8) {
        uint64_t v;
        memcpy(&v, pc.data() + off, 8);
        for (auto& [vb, db] : map)
            if (v >= vb.addr && v < vb.addr + vb.size) {
                v = db.va + (v - vb.addr);
                memcpy(pc.data() + off, &v, 8);
                ++moved;
                break;
            }
    }
    uint32_t pcd = uint32_t(pc.size() + 3) / 4;
    pc.resize(pcd * 4);
    auto dk = D.load_elf(onr::d3d::patch_elf(elf, pcd), onr::d3d::elf_threads(elf), pcd);
    printf("  -- %s (%s, %u windows), %u push constant dwords (%d addresses), ELF %zu bytes\n", st.kernel.c_str(), what,
           st.groups, pcd, moved, elf.size());

    auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    auto vk_rec = [&] {
        c.dev.begin();
        for (int i = 0; i < nd; ++i) { if (i) c.dev.barrier(); st.record(c); }
        return c.dev.submit();
    };
    auto dx_rec = [&] { return D.run(dk, pc.data(), st.groups, nd); };
    for (int r = 0; r < rounds; ++r) {
        double t[2];
        for (int a = 0; a < 2; ++a) {   // odd rounds start with D3D12
            int api = a ^ (r & 1);
            for (int w = 0; w < warm / nd; ++w) api ? dx_rec() : vk_rec();
            std::vector<double> all;
            for (int k = 0; k < c.reps; ++k) {
                auto ms = api ? dx_rec() : vk_rec();
                all.insert(all.end(), ms.begin(), ms.end());
            }
            t[api] = median(all);
        }
        printf("  round %d: median per dispatch  Vulkan %.4f ms   D3D12 %.4f ms   (D3D12 / Vulkan %.3f)\n", r, t[0], t[1], t[1] / t[0]);
    }

    int differ = 0;
    for (auto& [vb, db] : map) {
        std::vector<uint8_t> a(vb.size), b(vb.size);
        c.dev.download(vb, a.data(), vb.size);
        D.download(db, b.data(), vb.size);
        size_t eq = 0;
        for (size_t i = 0; i < a.size(); ++i) eq += a[i] == b[i];
        if (eq != a.size()) {
            ++differ;
            printf("  buffer of %llu bytes: %.6f%% bytes equal\n", (unsigned long long)vb.size, 100.0 * double(eq) / double(a.size()));
        }
        if (vb.buf == o.tensor.buf && cf.has_file("exp.u8")) {
            size_t n = size_t(o.size.x) * o.size.y * o.C;
            a.resize(n), b.resize(n);
            cmp_e4m3("Vulkan TensorOutput vs capture", a, cf.read<uint8_t>("exp.u8"));
            cmp_e4m3("D3D12  TensorOutput vs capture", b, cf.read<uint8_t>("exp.u8"));
        }
    }
    printf("  %zu buffers compared after both runs: %s\n", map.size(), differ ? "DIFFER" : "all byte-identical");
    for (auto& [vb, db] : map) D.destroy(db);
    D.destroy(dk);
    return differ;
}

int run_chain(Ctx& c, const fs::path& dir) {
    auto load = [&](const std::string& n) {
        CaseFile cf;
        if (!cf.load(dir / n)) throw std::runtime_error("missing case " + n);
        return cf;
    };
    Pool P(c);
    std::vector<Stage> st;
    std::vector<std::pair<CaseFile, Out*>> chk;
    Out enc[6], d0[5], d1[5], head;

    // ONR_CHAIN_START=<stage> (enc0..enc5, dec4_0 .. dec0_1): stages before it are not run; whatever they would
    // have produced comes from the capture (the case files of the consumers)
    const char* order[] = {"enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "dec4_0", "dec4_1", "dec3_0", "dec3_1",
                           "dec2_0", "dec2_1", "dec1_0", "dec1_1", "dec0_0", "dec0_1"};
    int start = 0;
    if (const char* s = getenv("ONR_CHAIN_START"))
        while (start < 15 && strcmp(order[start], s)) ++start;
    auto on = [&](const std::string& n) { int i = 0; while (n != order[i]) ++i; return i >= start; };
    if (start) printf("  chain starts at %s (earlier outputs from the capture)\n", order[start]);

    // launches 0 and 1 of the frame (DownsampledHistory2 and the exposure block feed rr2_enc0)
    CaseFile phc = load("pooled_f10"), aec = load("ae_f10");
    I2 pin{phc.i("in", 0), phc.i("in", 1)}, pout{phc.i("out", 0), phc.i("out", 1)};
    Buffer psrc = P.file(phc, "in.u32"), pdst = P.zero(size_t(pout.x) * pout.y * 4);
    PooledHistory2PC php{psrc.addr, pdst.addr, pin, pout};
    Buffer color = P.file(aec, "color.f16"), acc = P.zero(16), nacc = P.zero(1028), aexp = P.zero(4), prev = P.zero(4), blk = P.zero(16);
    uint32_t pv = uint32_t(aec.u("prev"));
    c.dev.upload(prev, &pv, 4);
    AeSumPC as{color.addr, acc.addr, {aec.i("size", 0), aec.i("size", 1)}, {aec.i("origin", 0), aec.i("origin", 1)}, {aec.i("max", 0), aec.i("max", 1)}};
    AeFinalPC af{acc.addr, nacc.addr, aexp.addr, prev.addr, blk.addr, aec.f("inv"), uint32_t(aec.i("reset")), 1028 / 4};
    I2 agrid{aec.i("grid", 0), aec.i("grid", 1)};

    // rr2_enc0: guides of the frame, histories of the previous frame as captured, our DownsampledHistory2 and exposure
    Enc0Out e0;
    CaseFile ce0;
    size_t n0 = 0;
    bool e0on = on("enc0");
    if (e0on) {
        ce0 = load("enc0_f10");
        Enc0In in = enc0_inputs(P, ce0);
        in.pm1 = P.file(ce0, "pm1.f16").addr, in.hidden = P.file(ce0, "hidden.u8").addr, in.ds = P.file(ce0, "ds.u32").addr;
        in.ds2 = pdst.addr, in.expo = blk.addr;
        for (auto& s : enc0_stages(P, ce0, in, e0)) st.push_back(s);
        n0 = st.size();
    }

    uint64_t x = 0, dimg = 0;
    for (int s = 1; s <= 5; ++s) {
        std::string n = "enc" + std::to_string(s);
        CaseFile cf = load(n + "_f10");
        if (!on(n)) continue;
        if (s == 1 && e0on) x = e0.net.tensor.addr, dimg = e0.net.dfn_pool.addr;
        else if (s == 1 || !on("enc" + std::to_string(s - 1))) x = P.file(cf, "x.u8").addr, dimg = P.file(cf, "dfn_in.f16").addr;
        st.push_back(enc_stage(P, cf, x, dimg, enc[s]));
        st.back().name = n;
        chk.push_back({cf, &enc[s]});
        x = enc[s].pooled.addr, dimg = enc[s].dfn_pool.addr;
    }
    uint64_t bt = enc[5].tensor.addr, bd = enc[5].dfn.addr;
    std::string lower = "enc5";
    for (int lv = 4; lv >= 0; --lv) {
        std::string n = "dec" + std::to_string(lv), e = "enc" + std::to_string(lv);
        CaseFile c0 = load(n + "_0_f10"), c1 = load(n + "_1_f10");
        bool enc_on = lv ? on(e) : e0on;
        Out& eo = lv ? enc[lv] : e0.net;
        if (on(n + "_0")) {
            st.push_back(dec_up(P, c0, enc_on ? eo.tensor.addr : P.file(c0, "x.u8").addr, on(lower) ? bt : P.file(c0, "xlo.u8").addr, d0[lv]));
            st.back().name = n + "_0";
            chk.push_back({c0, &d0[lv]});
        }
        if (on(n + "_1")) {
            uint64_t skip = enc_on ? eo.dfn.addr : P.file(c1, "skip.f16").addr;
            uint64_t xin = on(n + "_0") ? d0[lv].tensor.addr : P.file(c1, "x.u8").addr;
            st.push_back(dec_dfn(P, c1, xin, on(lower) ? bd : P.file(c1, "below.f16").addr, skip, lv ? d1[lv] : head));
            st.back().name = n + "_1";
            chk.push_back({c1, lv ? &d1[lv] : &head});
        }
        if (lv) bt = d1[lv].tensor.addr, bd = d1[lv].dfn.addr;
        lower = n + "_1";
    }
    for (auto& s : st)
        if (!c.k.has(s.kernel)) throw std::runtime_error("no kernel " + s.kernel);

    // post and downsample (whole frame) on our head outputs; InternalHistory from the capture, motion vectors and
    // exposure from our enc0 / auto-exposure when enc0 runs
    CaseFile pc = load("post_f10"), ds = load("ds_f10");
    I2 size{pc.i("size", 0), pc.i("size", 1)}, lr{pc.i("lr_size", 0), pc.i("lr_size", 1)}, dsz{pc.i("down_size", 0), pc.i("down_size", 1)};
    I2 in{ds.i("in_size", 0), ds.i("in_size", 1)}, out{ds.i("out_size", 0), ds.i("out_size", 1)};
    Buffer hist = P.file(pc, "hist.f16"), mv = P.file(pc, "mv.f16"), mve = P.file(pc, "mve.f16"), mvn = P.file(pc, "mvn.f16");
    uint64_t mva = e0on ? e0.pmv2.addr : mv.addr, mvea = e0on ? e0.ero.addr : mve.addr, mvna = e0on ? e0.nond.addr : mvn.addr;
    Buffer expo = e0on ? blk : P.file(pc, "exposure.f32"), bic = P.file(pc, "bicubic_gen.f32");
    Buffer tile = P.zero(size_t(lr.x) * lr.y * 24), hout = P.zero(size_t(size.x) * size.y * 8);
    Buffer down = P.zero(size_t(dsz.x) * dsz.y * 4), fix = P.zero(post_fix_bytes(size));
    Buffer frg = P.zero(size_t(size.x) * size.y * 4), fb = P.zero(size_t(size.x) * size.y * 2), dst = P.zero(size_t(out.x) * out.y * 8);
    PostTilePC tp{head.dfn.addr, head.ajab.addr, head.aniso.addr, tile.addr, lr, lr.x};
    const bool fused = fuse_tile(st, tile.addr);
    PostPC pp{};
    pp.tile = tile.addr, pp.mvf = head.mvf.addr, pp.hist_in = hist.addr, pp.mv = mva, pp.mv_eroded = mvea;
    pp.mv_nondil = mvna, pp.exposure = expo.addr, pp.bicubic = bic.addr, pp.hist_out = hout.addr;
    pp.out_rg = frg.addr, pp.out_b = fb.addr, pp.down = down.addr, pp.fix = fix.addr;
    pp.gate_scale = pc.f("gate_scale");
    for (int i = 0; i < 2; ++i)
        pp.scale[i] = pc.f("scale", i), pp.offset[i] = pc.f("offset", i), pp.inv_size[i] = pc.f("inv_size", i), pp.lr_inv_size[i] = pc.f("lr_inv_size", i);
    pp.size = size, pp.lr_size = lr, pp.origin = {0, 0}, pp.end = size, pp.slice = size, pp.reset = uint32_t(pc.i("reset"));
    Image hw_img;
    VkSampler hw_smp = VK_NULL_HANDLE;
    if (post_hw(c)) {
        hw_img = c.dev.alias_image(hist, uint32_t(size.x), uint32_t(size.y), VK_FORMAT_R16G16B16A16_SFLOAT, 8);
        hw_smp = c.dev.sampler(true, true);
        if (!hw_img.img) throw std::runtime_error("ONR_POST_HW: no linear image over the history");
        c.dev.set_texture(0, hw_img, hw_smp);
        pp.reset |= 16u;
    }
    DownsamplePC dp{frg.addr, fb.addr, dst.addr, expo.addr, in, out, {0, 0}, out, size, {0, size.x}};

    auto record = [&] {
        c.dev.fill(acc, 0, 16, 0);
        c.dev.barrier();
        c.k.pooled_history2(c.dev, php);
        c.k.auto_exposure(c.dev, as, af, agrid);
        c.dev.barrier();
        for (auto& s : st) {
            s.record(c);
            c.dev.barrier();
        }
        if (!fused) {
            c.k.post_tile(c.dev, tp);
            c.dev.barrier();
        }
        c.k.post(c.dev, pp, fix, sfx() == "_f");
        c.dev.barrier();
        c.k.downsample(c.dev, dp);
    };
    c.dev.begin();
    record();
    auto first = c.dev.submit();
    // ONR_WARM=n: n unmeasured recordings first, so the card reaches its game clock
    for (int r = 0, n = getenv("ONR_WARM") ? atoi(getenv("ONR_WARM")) : 0; r < n; ++r) {
        c.dev.begin();
        record();
        c.dev.submit();
    }
    std::vector<std::vector<double>> runs;
    std::vector<double> spans;
    for (int r = 0; r < c.reps; ++r) {
        c.dev.begin();
        record();
        runs.push_back(c.dev.submit());
        spans.push_back(c.dev.span());
    }
    std::vector<std::string> names = {"pooled_history2", "ae_sum", "ae_final"};
    for (auto& s : st) names.push_back(s.name + " (" + s.kernel + ")");
    for (const char* n : {"post_tile", "post", "post_fixup", "downsample"})
        if ((sfx() != "_f" || strcmp(n, "post_fixup")) && (!fused || strcmp(n, "post_tile"))) names.push_back(n);
    printf("  GPU time per dispatch (median of %d recordings):\n", c.reps);
    double total = 0, net = 0;
    for (size_t i = 0; i < names.size() && i < first.size(); ++i) {
        std::vector<double> t;
        for (auto& r : runs) t.push_back(r[i]);
        std::sort(t.begin(), t.end());
        double m = t[t.size() / 2];
        total += m;
        if (i >= 3 && i < 3 + st.size()) net += m;
        printf("    %-36s %.4f ms\n", names[i].c_str(), m);
    }
    std::sort(spans.begin(), spans.end());
    printf("  sum of dispatches %.3f ms (network enc0..dec0_1: %.3f ms), span of the recording %.3f ms\n", total, net, spans[spans.size() / 2]);
    if (getenv("ONR_VERBOSE"))
        for (auto& s : st) printf("    %-22s %s\n", s.kernel.c_str(), c.dev.stats(c.k[s.kernel]).c_str());

    int bad = 0;
    printf("  -- each stage against the capture (inputs: our previous stages)\n");
    if (e0on) bad += check_enc0(c, ce0, e0, "enc0");
    for (size_t i = 0; i < chk.size(); ++i) bad += check(c, chk[i].first, *chk[i].second, st[n0 + i].name);
    auto po = fetch<uint16_t>(c, dst, size_t(out.x) * out.y * 4);
    double f = cmp_f16("frame 10 final output RGB vs capture", po, ds.read<uint16_t>("exp_out.f16"), 4, 3);
    cmp_f16("InternalHistory vs capture", fetch<uint16_t>(c, hout, size_t(size.x) * size.y * 4), pc.read<uint16_t>("exp_hist.f16"));
    cmp_1010102("DownsampledHistory2 (launch 0) vs capture", fetch<uint32_t>(c, pdst, size_t(pout.x) * pout.y), phc.read<uint32_t>("exp_out.u32"));
    if (getenv("ONR_CHAIN_DUMP")) {
        FILE* fo = fopen((dir / "chain_out.f16").string().c_str(), "wb");
        fwrite(po.data(), 2, po.size(), fo);
        fclose(fo);
    }
    return bad + (f < 0.5);
}

// Frames 0..N-1 of the captured sequence with every recurrent state our own (InternalHistory, DownsampledHistory,
// HiddenHistory, PrevMvec, exposure); per frame only the guides and the launch parameters come from the capture.
// Each frame's final output is compared with the one NGX produced.
int run_seq(Ctx& c, const fs::path& dir) {
    auto load = [&](const fs::path& n) {
        CaseFile cf;
        if (!cf.load(dir / n)) throw std::runtime_error("missing case " + n.string());
        return cf;
    };
    std::vector<CaseFile> fr;
    for (int f = 0; f < 100; ++f) {
        char n[16];
        snprintf(n, sizeof n, "f%02d", f);
        CaseFile cf;
        if (!cf.load(dir / "seq" / n)) break;
        fr.push_back(cf);
    }
    if (const char* n = getenv("ONR_SEQ_FRAMES")) fr.resize(std::min(fr.size(), size_t(atoi(n))));
    if (fr.empty()) throw std::runtime_error("no seq/fNN cases (prep_cases.py seq)");
    Pool P(c);
    std::vector<Stage> st;
    std::vector<size_t> shifted;   // indices in st of the launches listed by "shifts" (enc1 .. dec0_1)

    // per-frame inputs: fixed buffers, re-uploaded every frame
    const char* gn[] = {"color.f16", "albedo.u8", "spec.u8", "normals.f16", "rough.f16", "depth.f32", "mv.f16", "hit.f16", "prm.f32"};
    std::vector<Buffer> gb;
    for (const char* g : gn) gb.push_back(P.file(fr[0], g));
    CaseFile ce0 = load("enc0_f10"), pc = load("post_f10"), ds = load("ds_f10"), phc = load("pooled_f10"), aec = load("ae_f10");
    I2 size{pc.i("size", 0), pc.i("size", 1)}, lr{pc.i("lr_size", 0), pc.i("lr_size", 1)}, dsz{pc.i("down_size", 0), pc.i("down_size", 1)};
    I2 in{ds.i("in_size", 0), ds.i("in_size", 1)}, out{ds.i("out_size", 0), ds.i("out_size", 1)};
    I2 pin{phc.i("in", 0), phc.i("in", 1)}, pout{phc.i("out", 0), phc.i("out", 1)};
    size_t lpx = size_t(lr.x) * lr.y, hbytes = size_t(size.x) * size.y * 8, dbytes = size_t(dsz.x) * dsz.y * 4;

    // recurrent state: two of each, ping-pong by frame parity (this frame writes [f & 1], reads [(f & 1) ^ 1])
    Buffer pm1 = P.zero(lpx * 8), hid_prev = P.zero(lpx * 4), ds_prev = P.zero(dbytes), hist_prev = P.zero(hbytes);
    Buffer pdst = P.zero(size_t(pout.x) * pout.y * 4), acc = P.zero(16), nacc = P.zero(1028), aexp = P.zero(4), prev = P.zero(4), blk = P.zero(16);
    Buffer tile = P.zero(lpx * 24), hout = P.zero(hbytes), down = P.zero(dbytes), fix = P.zero(post_fix_bytes(size));
    Buffer frg = P.zero(size_t(size.x) * size.y * 4), fb = P.zero(size_t(size.x) * size.y * 2), dst = P.zero(size_t(out.x) * out.y * 8);

    Enc0Out e0;
    Enc0In ein{gb[0].addr, gb[1].addr, gb[2].addr, gb[3].addr, gb[4].addr, gb[5].addr, gb[6].addr, gb[7].addr, pm1.addr, hid_prev.addr,
               ds_prev.addr, pdst.addr, blk.addr, gb[8].addr, {ce0.i("ds", 0), ce0.i("ds", 1)}, {ce0.i("ds", 2), ce0.i("ds", 3)}};
    for (auto& s : enc0_stages(P, ce0, ein, e0)) st.push_back(s);
    Out enc[6], d0[5], d1[5], head;
    uint64_t x = e0.net.tensor.addr, dimg = e0.net.dfn_pool.addr;
    for (int s = 1; s <= 5; ++s) {
        std::string n = "enc" + std::to_string(s);
        st.push_back(enc_stage(P, load(n + "_f10"), x, dimg, enc[s]));
        st.back().name = n;
        shifted.push_back(st.size() - 1);
        x = enc[s].pooled.addr, dimg = enc[s].dfn_pool.addr;
    }
    uint64_t bt = enc[5].tensor.addr, bd = enc[5].dfn.addr;
    for (int lv = 4; lv >= 0; --lv) {
        std::string n = "dec" + std::to_string(lv);
        Out& eo = lv ? enc[lv] : e0.net;
        st.push_back(dec_up(P, load(n + "_0_f10"), eo.tensor.addr, bt, d0[lv]));
        st.back().name = n + "_0";
        shifted.push_back(st.size() - 1);
        st.push_back(dec_dfn(P, load(n + "_1_f10"), d0[lv].tensor.addr, bd, eo.dfn.addr, lv ? d1[lv] : head));
        st.back().name = n + "_1";
        shifted.push_back(st.size() - 1);
        if (lv) bt = d1[lv].tensor.addr, bd = d1[lv].dfn.addr;
    }
    for (auto& s : st)
        if (!c.k.has(s.kernel)) throw std::runtime_error("no kernel " + s.kernel);

    PostTilePC tp{head.dfn.addr, head.ajab.addr, head.aniso.addr, tile.addr, lr, lr.x};
    const bool fused = fuse_tile(st, tile.addr);
    PostPC pp{};
    pp.tile = tile.addr, pp.mvf = head.mvf.addr, pp.hist_in = hist_prev.addr, pp.mv = e0.pmv2.addr, pp.mv_eroded = e0.ero.addr;
    pp.mv_nondil = e0.nond.addr, pp.exposure = blk.addr, pp.bicubic = P.file(pc, "bicubic_gen.f32").addr, pp.hist_out = hout.addr;
    pp.out_rg = frg.addr, pp.out_b = fb.addr, pp.down = down.addr, pp.fix = fix.addr;
    for (int i = 0; i < 2; ++i) pp.inv_size[i] = pc.f("inv_size", i), pp.lr_inv_size[i] = pc.f("lr_inv_size", i);
    pp.size = size, pp.lr_size = lr, pp.origin = {0, 0}, pp.end = size, pp.slice = size;
    DownsamplePC dp{frg.addr, fb.addr, dst.addr, blk.addr, in, out, {0, 0}, out, size, {0, size.x}};
    PooledHistory2PC php{ds_prev.addr, pdst.addr, pin, pout};
    AeSumPC as{gb[0].addr, acc.addr, {aec.i("size", 0), aec.i("size", 1)}, {aec.i("origin", 0), aec.i("origin", 1)}, {aec.i("max", 0), aec.i("max", 1)}};
    AeFinalPC af{acc.addr, nacc.addr, aexp.addr, prev.addr, blk.addr, aec.f("inv"), 0, 1028 / 4};
    I2 agrid{aec.i("grid", 0), aec.i("grid", 1)};

    auto set_shift = [](Stage& s, I2 sh) {
        auto* b = reinterpret_cast<EncBlockPC*>(s.pc.data());
        b->shift = sh;
        b->nwx = (b->size.x + sh.x + 7) / 8;
        s.groups = uint32_t(b->nwx * ((b->size.y + sh.y + 7) / 8));
    };
    std::vector<double> db, eq, ms;
    int bad = 0;
    printf("  %zu frames, network kernels %s\n", fr.size(), sfx().empty() ? "k8" : sfx().c_str());
    Buffer pmv2[2] = {e0.pmv2, pm1}, hid[2] = {head.hidden, hid_prev}, dsh[2] = {down, ds_prev}, hst[2] = {hout, hist_prev};
    Buffer ex[2] = {aexp, prev};
    VkSampler hw_smp = VK_NULL_HANDLE;
    Image hw_img[2];
    if (post_hw(c)) {
        hw_smp = c.dev.sampler(true, true);
        for (int i = 0; i < 2; ++i) {
            hw_img[i] = c.dev.alias_image(hst[i], uint32_t(size.x), uint32_t(size.y), VK_FORMAT_R16G16B16A16_SFLOAT, 8);
            if (!hw_img[i].img) throw std::runtime_error("ONR_POST_HW: no linear image over the history");
            c.dev.set_texture(uint32_t(i), hw_img[i], hw_smp);
        }
    }
    auto* ap = reinterpret_cast<Enc0AttrPC*>(st[0].pc.data());
    auto* hp = reinterpret_cast<DecDfnPC*>(st.back().pc.data());
    for (size_t f = 0; f < fr.size(); ++f) {
        const CaseFile& cf = fr[f];
        int cur = int(f & 1), old = cur ^ 1;
        ap->pmv2 = pmv2[cur].addr, ap->pm1 = pmv2[old].addr, ap->hidden = hid[old].addr, ap->ds = dsh[old].addr;
        hp->hidden = hid[cur].addr;
        pp.mv = pmv2[cur].addr, pp.hist_in = hst[old].addr, pp.hist_out = hst[cur].addr, pp.down = dsh[cur].addr;
        php.src = dsh[old].addr;
        af.exposure = ex[cur].addr, af.prev = ex[old].addr;
        if (f)
            for (size_t i = 0; i < gb.size(); ++i) {
                auto v = cf.read<uint8_t>(gn[i]);
                c.dev.upload(gb[i], v.data(), v.size());
            }
        else if (cf.kv.count("ae_prev")) {
            uint32_t v = uint32_t(cf.u("ae_prev"));
            c.dev.upload(ex[1], &v, 4);
        }
        reinterpret_cast<Enc0AttrPC*>(st[0].pc.data())->hist_off = uint32_t(cf.i("enc0", 1));
        reinterpret_cast<Enc0AttrPC*>(st[0].pc.data())->frame8 = uint32_t(cf.i("enc0", 0));
        reinterpret_cast<Enc0StagePC*>(st[1].pc.data())->frame8 = uint32_t(cf.i("enc0", 0));
        set_shift(st[1], {cf.i("shift", 0), cf.i("shift", 1)});
        for (size_t k = 0; k < shifted.size(); ++k) set_shift(st[shifted[k]], {cf.i("shifts", 2 * k), cf.i("shifts", 2 * k + 1)});
        af.reset = uint32_t(cf.i("ae_reset"));
        pp.gate_scale = cf.f("post_gate"), pp.reset = uint32_t(cf.i("post_reset"));
        if (hw_smp) pp.reset |= 16u | uint32_t(old) << 5;
        for (int i = 0; i < 2; ++i) pp.scale[i] = cf.f("post_scale", i), pp.offset[i] = cf.f("post_offset", i);

        c.dev.begin();
        c.dev.fill(acc, 0, 16, 0);
        c.dev.barrier();
        if (f) c.k.pooled_history2(c.dev, php);
        c.k.auto_exposure(c.dev, as, af, agrid);
        c.dev.barrier();
        for (auto& s : st) {
            s.record(c);
            c.dev.barrier();
        }
        if (!fused) {
            c.k.post_tile(c.dev, tp);
            c.dev.barrier();
        }
        c.k.post(c.dev, pp, fix, sfx() == "_f");
        c.dev.barrier();
        c.k.downsample(c.dev, dp);
        auto t = c.dev.submit();
        double sum = 0;
        for (double v : t) sum += v;
        ms.push_back(sum);

        auto po = fetch<uint16_t>(c, dst, size_t(out.x) * out.y * 4), xo = cf.read<uint16_t>("exp_out.f16");
        char what[64];
        snprintf(what, sizeof what, "frame %2zu final output RGB vs NGX", f);
        eq.push_back(cmp_f16(what, po, xo, 4, 3));
        double se = 0, peak = 0;
        size_t n = 0;
        for (size_t i = 0; i < po.size(); ++i) {
            if (i % 4 == 3) continue;
            double u = to_float(po[i]), v = to_float(xo[i]);
            se += (u - v) * (u - v), peak = std::max(peak, std::abs(v)), ++n;
        }
        db.push_back(se > 0 ? 10 * std::log10(peak * peak / (se / double(n))) : 999.0);
        if (getenv("ONR_SEQ_DUMP")) {
            char fn[32];
            snprintf(fn, sizeof fn, "seq_out_f%02zu.f16", f);
            FILE* fo = fopen((dir / fn).string().c_str(), "wb");
            fwrite(po.data(), 2, po.size(), fo);
            fclose(fo);
        }
    }
    printf("  frame  PSNR dB  equal %%   GPU ms\n");
    for (size_t f = 0; f < db.size(); ++f) printf("  %5zu  %7.2f  %7.3f  %7.2f\n", f, db[f], 100 * eq[f], ms[f]);
    for (double v : db) bad += v < 45.0;
    return bad;
}

// rr2_enc0's optional guides (docs/NOTA-ENTRADAS.md) on the enc0_f10 case, against nvngx_dlssd.dll 310.9.1 as read from
// its PTX: synthetic DLSSD.ResponsivityMask, DLSSD.ColorBeforeTransparency and DLSSD.DepthOfFieldGuide must leave every
// other attribute as it was and give attributes 33, 30 and 32 by the reference's formulas; GBuffer.SpecularMvec must
// give PrevMvec2.zw = (m + offset) * scale; without a hit distance PrevMvec2.zw must be the pixel's own vector. Then the
// fast halves: the 64-byte record with zero guides must give the 48-byte record's tensor bit for bit, and the
// reflection vectors given as the motion vectors themselves must give the no-hit-distance frame bit for bit.
int run_optguides(Ctx& c, const fs::path& cases) {
    CaseFile cf;
    if (!cf.load(cases / "enc0_f10")) throw std::runtime_error("no enc0_f10 case");
    Pool P(c);
    Enc0In in = enc0_inputs(P, cf);
    in.pm1 = P.file(cf, "pm1.f16").addr, in.hidden = P.file(cf, "hidden.u8").addr, in.ds = P.file(cf, "ds.u32").addr;
    in.ds2 = P.file(cf, "ds2.u32").addr, in.expo = P.file(cf, "expo.f32").addr;
    const size_t px = size_t(1280) * 720;
    const auto prm = cf.read<uint8_t>("prm.f32");
    const auto color = cf.read<uint16_t>("color.f16");
    const float E = cf.read<float>("expo.f32").at(0);
    auto C = [&](uint32_t cb) { float v; memcpy(&v, prm.data() + (cb - 0x160), 4); return v; };
    auto f2h = [](float f) {   // round to nearest even
        uint32_t x = fbits(f), s = (x >> 16) & 0x8000u, e = (x >> 23) & 255u, m = x & 0x7fffffu;
        if (e == 255) return uint16_t(s | 0x7c00u | (m ? 0x200u : 0u));
        int ee = int(e) - 127 + 15;
        if (ee >= 31) return uint16_t(s | 0x7c00u);
        if (ee <= 0) {
            if (ee < -10) return uint16_t(s);
            m |= 0x800000u;
            uint32_t sh = uint32_t(14 - ee), r = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
            if (rem > half || (rem == half && (r & 1))) ++r;
            return uint16_t(s | r);
        }
        uint32_t r = uint32_t(ee) << 10 | m >> 13, rem = m & 0x1fffu;
        if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
        return uint16_t(s | r);
    };
    auto hash = [](size_t i) { uint32_t h = uint32_t(i) * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; return h; };
    // the guides: values across the clamps, NaN and inf here and there
    std::vector<float> resp(px), dof(px);
    std::vector<uint16_t> cbt(px * 4), smv(px * 2);
    for (size_t i = 0; i < px; ++i) {
        const uint32_t h = hash(i);
        resp[i] = i % 997 == 5 ? NAN : i % 991 == 7 ? INFINITY : float(int(h % 4001) - 2000) / 150.0f;
        dof[i] = i % 983 == 3 ? NAN : float(int((h >> 8) % 20001) - 10000) / 700.0f;
        const float k = float((h >> 4) % 301) / 100.0f;
        for (int ch = 0; ch < 4; ++ch) cbt[i * 4 + ch] = f2h(to_float(color[i * 4 + ch]) * k + (ch == 1 ? 0.01f : 0.0f));
        smv[i * 2] = f2h(float(int(h % 2001) - 1000) / 37.0f), smv[i * 2 + 1] = f2h(float(int((h >> 11) % 2001) - 1000) / 53.0f);
    }
    const Buffer b_resp = P.vec(resp), b_dof = P.vec(dof), b_cbt = P.vec(cbt), b_smv = P.vec(smv);
    const Buffer z_resp = P.zero(px * 4), z_dof = P.zero(px * 4);
    // one enc0 run: the parameter block with the table t after it, the attribute half's kernel and the stage's (sk empty:
    // the attribute half alone)
    struct Run { std::vector<uint16_t> attr, pmv2, nond; std::vector<uint8_t> tensor; };
    auto run = [&](const uint64_t* t, const std::string& ak, const std::string& sk, bool wide) {
        std::vector<uint8_t> blk(prm);
        blk.resize(1088 + 64, 0);
        if (t) memcpy(blk.data() + 1088, t, 40);
        Enc0In i2 = in;
        i2.prm = P.vec(blk).addr;
        Enc0Out o;
        auto st = enc0_stages(P, cf, i2, o);
        auto* a = reinterpret_cast<Enc0AttrPC*>(st[0].pc.data());
        if (t) a->hist_off |= 2u;
        if (wide) {   // the 64-byte record: its own buffer (enc0_stages sized the 48-fp16 one)
            const Buffer w = P.zero(size_t(1280) * 736 * 64 + 64);
            a->attr = w.addr;
            memcpy(st[1].pc.data() + offsetof(Enc0StagePC, attr), &w.addr, 8);
        }
        st[0].kernel = ak, st[1].kernel = sk;
        if (!c.k.has(ak) || (!sk.empty() && !c.k.has(sk))) throw std::runtime_error("no kernel " + ak + " / " + sk);
        time_it(c, [&] {
            st[0].record(c);
            c.dev.barrier();
            if (!sk.empty()) st[1].record(c);
        });
        Run r;
        if (sk.empty()) r.attr = fetch<uint16_t>(c, o.attr, px * 48);
        r.pmv2 = fetch<uint16_t>(c, o.pmv2, px * 4), r.nond = fetch<uint16_t>(c, o.nond, px * 2);
        if (!sk.empty()) r.tensor = fetch<uint8_t>(c, o.net.tensor, size_t(1280) * 736 * 32);
        return r;
    };
    int bad = 0;
    auto ulp = [](uint16_t a, uint16_t b) {
        auto ord = [](uint16_t h) { return (h & 0x8000u) ? -int(h & 0x7fffu) : int(h); };
        return std::abs(ord(a) - ord(b));
    };
    const Run r0 = run(nullptr, "enc0_attr", "", false);
    {
        const uint64_t t[5] = {b_resp.addr, b_cbt.addr, b_dof.addr, 0, 0};
        const Run r1 = run(t, "enc0_attr", "", false);
        size_t other = 0, eq[3] = {}, near[3] = {}, n = 0;
        int worst[3] = {};
        auto slog = [](float v) {
            float s = (std::log2(std::fma(std::fabs(v), 100.0f, 1.0f)) * 0.30103000998497009277f) * (v >= 0.0f ? 0.36000001430511474609f : -0.36000001430511474609f);
            return std::isnan(s) ? 0.0f : std::clamp(s, -8.0f, 8.0f);
        };
        for (size_t i = 0; i < px; ++i) {
            for (int k = 0; k < 48; ++k)
                if (k != 30 && k != 32 && k != 33) other += r1.attr[i * 48 + k] != r0.attr[i * 48 + k];
            float d[3];
            for (int ch = 0; ch < 3; ++ch) {
                float e = E * to_float(color[i * 4 + ch]);
                e = std::isnan(e) ? 0.0f : std::min(e, 65504.0f);
                d[ch] = e - E * to_float(cbt[i * 4 + ch]);
            }
            const float rr = std::fma(resp[i], C(0x3ac), C(0x3b0));
            const uint16_t want[3] = {f2h(slog(std::fma(d[2], 0.25f, std::fma(d[0], 0.25f, d[1] * 0.5f)))), f2h(slog(E * dof[i])),
                                      f2h(std::isnan(rr) ? 0.0f : std::clamp(rr, -8.0f, 8.0f))};
            const int idx[3] = {30, 32, 33};
            for (int k = 0; k < 3; ++k) {
                const int u = ulp(r1.attr[i * 48 + idx[k]], want[k]);
                eq[k] += u == 0, near[k] += u <= 1, worst[k] = std::max(worst[k], u);
            }
            ++n;
        }
        printf("  responsivity + ColorBeforeTransparency + depth of field: other attributes changed in %zu values (want 0)\n", other);
        const char* nm[3] = {"30 ColorBeforeTransparency", "32 depth of field guide", "33 responsivity mask"};
        for (int k = 0; k < 3; ++k)
            printf("  attribute %-28s equal %.4f%%, within 1 fp16 ulp %.4f%%, worst %d ulp\n", nm[k], 100.0 * double(eq[k]) / double(n),
                   100.0 * double(near[k]) / double(n), worst[k]);
        bad += other != 0 || worst[2] != 0 || worst[0] > 1 || worst[1] > 1;
        printf("  p588 %g, p592 %g (block of the capture)\n", C(0x3ac), C(0x3b0));
    }
    {
        const uint64_t t[5] = {0, 0, 0, b_smv.addr, 0};
        const Run r2 = run(t, "enc0_attr", "", false);
        size_t zw = 0, xy = 0;
        for (size_t i = 0; i < px; ++i) {
            const float x = (to_float(smv[i * 2]) + C(0x328)) * C(0x320), y = (to_float(smv[i * 2 + 1]) + C(0x32c)) * C(0x324);
            zw += r2.pmv2[i * 4 + 2] != f2h(x) || r2.pmv2[i * 4 + 3] != f2h(y);
            xy += r2.pmv2[i * 4] != r0.pmv2[i * 4] || r2.pmv2[i * 4 + 1] != r0.pmv2[i * 4 + 1];
        }
        printf("  GBuffer.SpecularMvec: PrevMvec2.zw off the formula in %zu pixels, PrevMvec2.xy changed in %zu (want 0, 0)\n", zw, xy);
        bad += zw != 0 || xy != 0;
        const uint64_t t3[5] = {0, 0, 0, 0, 1};
        const Run r3 = run(t3, "enc0_attr", "", false);
        size_t own = 0, moved = 0;
        for (size_t i = 0; i < px; ++i) {
            own += r3.pmv2[i * 4 + 2] != r3.nond[i * 2] || r3.pmv2[i * 4 + 3] != r3.nond[i * 2 + 1];
            moved += r3.pmv2[i * 4 + 2] != r0.pmv2[i * 4 + 2] || r3.pmv2[i * 4 + 3] != r0.pmv2[i * 4 + 3];
        }
        printf("  no hit distance: PrevMvec2.zw != the pixel's vector in %zu pixels (want 0); differs from the derived one in %zu\n", own, moved);
        bad += own != 0 || moved == 0;
    }
    if (c.k.has("enc0_attr_f_w") && c.k.has("enc0_stage_32_64_f_w")) {
        const Run n0 = run(nullptr, "enc0_attr_f", "enc0_stage_32_64_f", false);
        const uint64_t tz[5] = {z_resp.addr, 0, z_dof.addr, 0, 0};
        const Run w0 = run(tz, "enc0_attr_f_w", "enc0_stage_32_64_f_w", true);
        const uint64_t tg[5] = {b_resp.addr, 0, b_dof.addr, 0, 0};
        const Run w1 = run(tg, "enc0_attr_f_w", "enc0_stage_32_64_f_w", true);
        size_t d0 = 0, d1 = 0;
        for (size_t i = 0; i < n0.tensor.size(); ++i) d0 += w0.tensor[i] != n0.tensor[i], d1 += w1.tensor[i] != n0.tensor[i];
        printf("  fast, 64-byte record with a zero mask and guide: TensorOutput differs from the 48-byte one in %zu bytes (want 0)\n", d0);
        printf("  fast, 64-byte record with the mask and the guide: TensorOutput differs in %zu of %zu bytes (they reach the embed)\n", d1,
               n0.tensor.size());
        bad += d0 != 0 || d1 == 0;
        const uint64_t ts[5] = {0, 0, 0, in.mv, 0}, th[5] = {0, 0, 0, 0, 1};
        const Run s = run(ts, "enc0_attr_f", "enc0_stage_32_64_f", false), h = run(th, "enc0_attr_f", "enc0_stage_32_64_f", false);
        size_t ds = 0;
        for (size_t i = 0; i < s.tensor.size(); ++i) ds += s.tensor[i] != h.tensor[i];
        for (size_t i = 0; i < px * 4; ++i) ds += s.pmv2[i] != h.pmv2[i];
        printf("  fast, SpecularMvec = the motion vectors against no hit distance: %zu values differ (want 0)\n", ds);
        bad += ds != 0;
    } else {
        printf("  no 64-byte record kernels (enc0_attr_f_w, enc0_stage_32_64_f_w)\n");
        ++bad;
    }
    printf(bad ? "  optional guides: %d check(s) failed\n" : "  optional guides: all checks pass\n", bad);
    return bad;
}
