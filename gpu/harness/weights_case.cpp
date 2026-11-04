// SPDX-License-Identifier: MIT
// Weights rebuilt in C++ from the DLL (weights.cpp) against the case files written by prep_cases.py from the
// Python reference (ref-impl/weights.py): every tensor and kernel scalar must be bit for bit the same.
#include <cstdio>
#include <cstring>

#include "harness.hpp"
#include "opennr_rr/weights.hpp"

using namespace onr;

namespace {

int arena_channel(int c) {
    int g = c & 31;
    return (c & ~31) + ((g >> 3) & 1) * 16 + ((g >> 4) & 1) * 2 + 4 * ((g & 7) >> 1) + (g & 1);
}
H16 perm_k(const H16& w, int K) {
    H16 r(w.size());
    for (size_t n = 0; n < w.size() / K; ++n)
        for (int k = 0; k < K; ++k) r[n * K + k] = w[n * K + arena_channel(k)];
    return r;
}
H16 perm_n(const H16& w, int N) {
    size_t K = w.size() / N;
    H16 r(w.size());
    for (int n = 0; n < N; ++n)
        for (size_t k = 0; k < K; ++k) r[n * K + k] = w[size_t(arena_channel(n)) * K + k];
    return r;
}

int bad = 0;
void same(const char* tag, const char* what, const H16& ours, const H16& ref) {
    size_t n = std::min(ours.size(), ref.size()), d = 0;
    for (size_t i = 0; i < n; ++i) d += ours[i] != ref[i];
    bool ok = ours.size() == ref.size() && d == 0;
    if (!ok) printf("  %-10s %-14s DIFFERS: %zu of %zu (sizes %zu / %zu)\n", tag, what, d, n, ours.size(), ref.size());
    bad += !ok;
}
void same_u(const char* tag, const char* what, uint32_t ours, uint32_t ref) {
    if (ours != ref) printf("  %-10s %-14s DIFFERS: 0x%08x / 0x%08x\n", tag, what, ours, ref), ++bad;
}

void block(const char* tag, const CaseFile& cf, const BlockWeights& b) {
    int C = cf.i("dims", 0);
    auto w = [&](const char* f) { return cf.read<uint16_t>(f); };
    same(tag, "in_a", b.in_a, perm_k(w("w_attn_in_a.f16"), C));
    same(tag, "in_b", b.in_b, perm_k(w("w_attn_in_b.f16"), C));
    same(tag, "out", b.out, perm_n(w("w_attn_out.f16"), C));
    same(tag, "out_b", b.out_b, perm_n(w("w_attn_out_b.f16"), C));
    same(tag, "fc1", b.fc1, perm_k(w("w_mlp_fc1.f16"), C));
    same(tag, "fc1_b", b.fc1_b, w("w_mlp_fc1_b.f16"));
    same(tag, "fc2", b.fc2, perm_n(w("w_mlp_fc2.f16"), C));
    same(tag, "fc2_b", b.fc2_b, perm_n(w("w_mlp_fc2_b.f16"), C));
    same(tag, "bias", b.bias, w("w_attn_bias.f16"));
    same_u(tag, "ls", b.ls, uint32_t(cf.u("ls", 0)) | uint32_t(cf.u("ls", 1)) << 16);
}

uint32_t hex(const CaseFile& cf, const char* k, size_t n) { return uint32_t(std::stoul(cf.get(k).at(n), nullptr, 16)); }

}  // namespace

int run_weights(const fs::path& dll, const fs::path& dir) {
    Weights w;
    std::string err;
    if (!build_weights(dll.wstring(), w, err)) {
        printf("  build_weights failed: %s\n", err.c_str());
        return 1;
    }
    printf("  weights built from %s (version %s)\n", dll.string().c_str(), w.version.c_str());
    auto load = [&](const std::string& n) {
        CaseFile cf;
        if (!cf.load(dir / n)) throw std::runtime_error("missing case " + n);
        return cf;
    };
    {
        CaseFile cf = load("enc0_f10");
        const StageWeights& s = w.enc[0];
        block("enc0", cf, s.b);
        same("enc0", "dfn_w", s.dfn_w, cf.read<uint16_t>("w_stage_dfn.f16"));
        same("enc0", "dfn_b", s.dfn_b, cf.read<uint16_t>("w_stage_dfn_b.f16"));
        same("enc0", "embed_w", w.embed_w, cf.read<uint16_t>("w_embed.f16"));
        same("enc0", "embed_b", w.embed_b, cf.read<uint16_t>("w_embed_b.f16"));
        same("enc0", "lut", w.lut, cf.read<uint16_t>("lut.u16"));
        same_u("enc0", "s_dfn", s.s_dfn, hex(cf, "stage", 1));
        auto prm = cf.read<uint32_t>("prm.f32");
        for (int i = 0; i < 4; ++i) {
            uint32_t b;
            memcpy(&b, &w.enc0_scalars[i], 4);
            same_u("enc0", ("prm " + std::to_string(572 + 4 * i)).c_str(), b, prm[(572 + 4 * i) / 4]);
        }
    }
    for (int s = 1; s <= 5; ++s) {
        std::string n = "enc" + std::to_string(s);
        CaseFile cf = load(n + "_f10");
        const StageWeights& st = w.enc[s];
        block(n.c_str(), cf, st.b);
        same(n.c_str(), "dfn_w", st.dfn_w, cf.read<uint16_t>("w_stage_dfn.f16"));
        same(n.c_str(), "dfn_b", st.dfn_b, cf.read<uint16_t>("w_stage_dfn_b.f16"));
        H16 mw = cf.i("stage", 0) ? cf.read<uint16_t>("w_stage_merge.f16") : H16(16, 0);
        H16 mb = cf.i("stage", 0) ? cf.read<uint16_t>("w_stage_merge_b.f16") : H16(16, 0);
        if (s == 1) {
            auto a = cf.read<uint16_t>("w_merge_in.f16"), b = cf.read<uint16_t>("w_merge_in_b.f16");
            mw.insert(mw.end(), a.begin(), a.end());
            mb.insert(mb.end(), b.begin(), b.end());
        }
        same(n.c_str(), "mrg_w", st.mrg_w, mw);
        same(n.c_str(), "mrg_b", st.mrg_b, mb);
        same_u(n.c_str(), "s_dfn", st.s_dfn, hex(cf, "stage", 1));
        same_u(n.c_str(), "Co", uint32_t(st.Co), uint32_t(cf.i("stage", 0)));
    }
    for (int lv = 4; lv >= 0; --lv) {
        std::string n = "dec" + std::to_string(lv);
        CaseFile c0 = load(n + "_0_f10"), c1 = load(n + "_1_f10");
        const StageWeights &a = w.dec0[lv], &b = w.dec1[lv];
        block((n + "_0").c_str(), c0, a.b);
        same((n + "_0").c_str(), "up_w", a.up_w, c0.read<uint16_t>("w_up.f16"));
        same((n + "_0").c_str(), "up_b", a.up_b, c0.read<uint16_t>("w_up_b.f16"));
        same_u((n + "_0").c_str(), "s98", a.s98, hex(c0, "dec0", 1));
        same_u((n + "_0").c_str(), "Cp", uint32_t(a.Cp), uint32_t(c0.i("dec0", 0)));
        block((n + "_1").c_str(), c1, b.b);
        if (lv) {
            same((n + "_1").c_str(), "dfn_w", b.dfn_w, c1.read<uint16_t>("w_dfn.f16"));
            same((n + "_1").c_str(), "dfn_b", b.dfn_b, c1.read<uint16_t>("w_dfn_b.f16"));
            same_u((n + "_1").c_str(), "s_dfn", b.s_dfn, hex(c1, "dec1", 0));
        } else {
            same_u("dec0_1", "s_dfn", b.s_dfn, hex(c1, "head", 0));
            uint32_t ks, kt;
            memcpy(&ks, &w.head_k_scale, 4);
            memcpy(&kt, &w.head_k_tanh, 4);
            same_u("dec0_1", "k_scale", ks, fbits(c1.f("head", 1)));
            same_u("dec0_1", "k_tanh", kt, fbits(c1.f("head", 2)));
        }
    }
    {
        CaseFile cf = load("post_f10");
        uint32_t g;
        memcpy(&g, &w.post_gate, 4);
        same_u("post", "gate_scale", g, fbits(cf.f("gate_scale")));
    }
    printf("  weights: %s\n", bad ? "MISMATCH" : "every tensor and scalar bit for bit equal to the Python reference");
    return bad;
}
