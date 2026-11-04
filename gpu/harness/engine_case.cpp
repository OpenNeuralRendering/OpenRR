// SPDX-License-Identifier: MIT
// The runtime's Engine (weights from the DLL, every launch parameter derived from the frame's inputs) on the
// captured sequence: the derived rr2_enc0 block against the captured one (fields the kernels read), and each
// frame's final output against NGX. The inputs per frame are the scene's (jitter, matrices: identity view, the
// projection that the capture's block carries), nothing else from the capture.
#include <cmath>
#include <cstdio>
#include <cstring>

#include "harness.hpp"
#include "opennr_rr/engine.hpp"

using namespace onr;

namespace {

// constant-bank addresses the enc0 kernels read (enc0_attr.comp), as (first, count of words)
const std::pair<uint32_t, int> kUsed[] = {{0x2e8, 18}, {0x350, 6}, {0x39c, 4}, {0x3d8, 3}, {0x424, 16}, {0x4e4, 16}, {0x524, 16}, {0x590, 1}};

double psnr_rgb(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    double se = 0, peak = 0;
    size_t n = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        if (i % 4 == 3) continue;
        double u = to_float(a[i]), v = to_float(b[i]);
        se += (u - v) * (u - v), peak = std::max(peak, std::abs(v)), ++n;
    }
    return se > 0 ? 10 * std::log10(peak * peak / (se / double(n))) : 999.0;
}

uint64_t fnv(const void* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const uint8_t*>(p)[i]) * 1099511628211ull;
    return h;
}

}  // namespace

int run_engine(Ctx& c, const fs::path& dll, const fs::path& dir) {
    Weights w;
    std::string err;
    if (!build_weights(dll.wstring(), w, err)) throw std::runtime_error("weights: " + err);
    bool fast = getenv("ONR_F32") != nullptr;
    c.k.load(c.dev, ONR_SPV_DIR, Engine::kernel_names(w, fast));
    Engine e;
    e.init(c.dev, c.k, w);
    e.set_fast(fast);
    std::vector<CaseFile> fr;
    for (int f = 0; f < 100; ++f) {
        char n[16];
        snprintf(n, sizeof n, "f%02d", f);
        CaseFile cf;
        if (!cf.load(dir / "seq" / n)) break;
        fr.push_back(cf);
    }
    if (fr.empty()) throw std::runtime_error("no seq/fNN cases (prep_cases.py seq)");
    const char* gn[] = {"color.f16", "albedo.u8", "spec.u8", "normals.f16", "rough.f16", "depth.f32", "mv.f16", "hit.f16"};
    std::vector<Buffer> gb;
    for (const char* g : gn) {
        auto v = fr[0].read<uint8_t>(g);
        gb.push_back(c.dev.create(v.size()));
    }
    I2 render{1280, 720}, output{1920, 1080};
    Buffer out = c.dev.create(size_t(output.x) * output.y * 8);
    Guides g{gb[0].addr, gb[1].addr, gb[2].addr, gb[3].addr, gb[4].addr, gb[5].addr, gb[6].addr, gb[7].addr};
    e.resize(render, output);
    int bad = 0;
    size_t block_diff = 0;
    std::map<uint32_t, int> diff_at;
    printf("  %zu frames, network kernels %s\n  frame  PSNR dB   GPU ms  output hash\n", fr.size(), fast ? "_f (fast)" : "k8 (faithful)");
    for (size_t f = 0; f < fr.size(); ++f) {
        const CaseFile& cf = fr[f];
        for (size_t i = 0; i < gb.size(); ++i) {
            auto v = cf.read<uint8_t>(gn[i]);
            c.dev.upload(gb[i], v.data(), v.size());
        }
        auto prm = cf.read<uint32_t>("prm.f32");
        EngineFrame ef;
        ef.render = render, ef.output = output;
        memcpy(&ef.jitter[0], &prm[(0x2f0 - 0x160) / 4], 4);
        memcpy(&ef.jitter[1], &prm[(0x2f4 - 0x160) / 4], 4);
        for (int i = 0; i < 16; ++i) {
            memcpy(&ef.view_to_clip[i], &prm[(0x424 - 0x160) / 4 + i], 4);
            ef.world_to_view[i] = i % 5 == 0 ? 1.0f : 0.0f;
        }
        ef.reset = f == 0;
        c.dev.begin();
        e.record(ef, g, out.addr);
        auto t = c.dev.submit();
        double ms = 0;
        for (double v : t) ms += v;
        // the block the engine derived for this frame, against the captured one
        std::vector<uint32_t> ours(272);
        {
            float pj[2] = {0, 0};
            if (f) {
                auto pp = fr[f - 1].read<uint32_t>("prm.f32");
                memcpy(&pj[0], &pp[(0x2f0 - 0x160) / 4], 4), memcpy(&pj[1], &pp[(0x2f4 - 0x160) / 4], 4);
            }
            Enc0BlockInput bi{render, {2880, 1620}, {1440, 810}, {ef.jitter[0], ef.jitter[1]}, {pj[0], pj[1]}, {1, 1},
                              ef.world_to_view, ef.view_to_clip, ef.world_to_view, ef.view_to_clip, w.enc0_scalars};
            auto b = enc0_block(bi);
            ours.assign(b.begin(), b.end());
        }
        for (auto [cb, n] : kUsed)
            for (int i = 0; i < n; ++i) {
                uint32_t k = (cb - 0x160) / 4 + i;
                if (ours[k] != prm[k]) {
                    ++diff_at[cb + 4 * i];
                    if (block_diff < 6) {
                        float a, b;
                        memcpy(&a, &ours[k], 4), memcpy(&b, &prm[k], 4);
                        printf("    frame %zu block 0x%03x: ours %08x (%g) capture %08x (%g)\n", f, cb + 4 * i, ours[k], a, prm[k], b);
                    }
                    ++block_diff;
                }
            }
        std::vector<uint16_t> po(size_t(output.x) * output.y * 4);
        c.dev.download(out, po.data(), po.size() * 2);
        double db = psnr_rgb(po, cf.read<uint16_t>("exp_out.f16"));
        printf("  %5zu  %7.2f  %7.2f  %016llx\n", f, db, ms, (unsigned long long)fnv(po.data(), po.size() * 2));
        bad += db < 45.0;
    }
    printf("  enc0 block: %zu used words differ from the capture over the sequence\n", block_diff);
    for (auto& b : gb) c.dev.destroy(b);
    c.dev.destroy(out);
    e.destroy();
    return bad;
}
