// SPDX-License-Identifier: MIT
// The denoised-image hop of FASE5-NR.md on the captured sequence: the Engine recorded whole (record) and in two
// halves with the image exported and imported between them, with no edit and with a constant tint, in the loop and
// display modes. Checks, every frame:
//   no edit, either mode   output, InternalHistory and DownsampledHistory bit-identical to the whole record
//   tint, display          both histories bit-identical to the whole record, output different
//   tint, loop             histories and output different
//   tint, display from frame 6 (and again with frame 9 without an edit: the host refused it): histories
//                          bit-identical, output equal before frame 6 only
// and how much of the tint reaches the output per frame (display adds the edit as a delta: ~1 from its first frame)
// and prints the GPU time of each half (span of the submit).
#include <cmath>
#include <cstdio>
#include <cstring>

#include "harness.hpp"
#include "opennr_rr/engine.hpp"

using namespace onr;

namespace {

uint64_t fnv(const void* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const uint8_t*>(p)[i]) * 1099511628211ull;
    return h;
}

struct FrameHashes { uint64_t out, hist, down; double green; };   // green: sum of the output's G

struct Run {
    const char* name;
    bool split;
    DfnEdit edit;
    bool tint;
    int from = 0, refused = -1;   // display from frame `from` (no hop before), no edit on frame `refused`
    std::vector<FrameHashes> h;
    double net_ms = 0, post_ms = 0, export_ms = 0;
    std::vector<double> post_each;   // per dispatch of the post half
    int timed = 0;
};

}  // namespace

int run_nrdfn(Ctx& c, const fs::path& dll, const fs::path& dir) {
    Weights w;
    std::string err;
    if (!build_weights(dll.wstring(), w, err)) throw std::runtime_error("weights: " + err);
    bool fast = getenv("ONR_F32") != nullptr;
    c.k.load(c.dev, ONR_SPV_DIR, Engine::kernel_names(w, fast));
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
    for (const char* g : gn) gb.push_back(c.dev.create(fr[0].read<uint8_t>(g).size()));
    const I2 render{1280, 720}, output{1920, 1080};
    const uint32_t pitch = (uint32_t(render.x) * 16 + 255) & ~255u;
    const size_t io_bytes = size_t(pitch) * render.y;
    Buffer out = c.dev.create(size_t(output.x) * output.y * 8), io = c.dev.create(io_bytes);
    Guides g{gb[0].addr, gb[1].addr, gb[2].addr, gb[3].addr, gb[4].addr, gb[5].addr, gb[6].addr, gb[7].addr};
    const float tint[3] = {1.0f, 0.6f, 0.6f};

    std::vector<Run> runs = {{"whole record", false, DfnEdit::none, false, {}},
                             {"split, no edit, loop", true, DfnEdit::loop, false, {}},
                             {"split, no edit, display", true, DfnEdit::display, false, {}},
                             {"split, tint, loop", true, DfnEdit::loop, true, {}},
                             {"split, tint, display", true, DfnEdit::display, true, {}},
                             {"split, off (no export)", true, DfnEdit::none, false, {}},
                             {"split, tint, display f6+", true, DfnEdit::display, true, 6, -1, {}},
                             {"split, tint, display f6+ r9", true, DfnEdit::display, true, 6, 9, {}}};
    std::vector<uint16_t> po(size_t(output.x) * output.y * 4);
    std::vector<float> img(io_bytes / 4);
    const int warm = getenv("ONR_WARM") ? atoi(getenv("ONR_WARM")) : 0;
    for (Run& r : runs) {
        if (warm) {   // ONR_WARM=n: n unmeasured recordings of frame 0 on a scratch engine, so the card is at its game clock
            for (size_t i = 0; i < gb.size(); ++i) {
                auto v = fr[0].read<uint8_t>(gn[i]);
                c.dev.upload(gb[i], v.data(), v.size());
            }
            Engine s;
            s.init(c.dev, c.k, w);
            s.set_fast(fast);
            s.resize(render, output);
            EngineFrame ef;
            ef.render = render, ef.output = output;
            for (int i = 0; i < warm; ++i) {
                c.dev.begin();
                s.record(ef, g, out.addr);
                c.dev.submit();
            }
            s.destroy();
        }
        Engine e;
        e.init(c.dev, c.k, w);
        e.set_fast(fast);
        if (const char* a = getenv("ONR_DELTA_ALPHA")) e.set_delta_alpha(float(atof(a)));
        e.resize(render, output);
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
            double net = 0, post = 0;
            if (!r.split) {
                c.dev.begin();
                e.record(ef, g, out.addr);
                c.dev.submit();
                net = c.dev.span();
            } else {
                const bool hop = r.edit != DfnEdit::none && int(f) >= r.from;
                const bool edited = hop && int(f) != r.refused;
                c.dev.begin();
                e.record_net(ef, g, hop ? io.addr : 0, pitch, hop ? r.edit : DfnEdit::none);
                auto t1 = c.dev.submit();
                net = c.dev.span();
                if (f >= 2 && hop) r.export_ms += t1.back();
                if (r.tint && edited) {
                    c.dev.download(io, img.data(), io_bytes);
                    for (int y = 0; y < render.y; ++y)
                        for (int x = 0; x < render.x; ++x)
                            for (int k = 0; k < 3; ++k) img[(size_t(y) * pitch / 4) + size_t(x) * 4 + k] *= tint[k];
                    c.dev.upload(io, img.data(), io_bytes);
                }
                c.dev.begin();
                e.record_post(ef, out.addr, hop ? r.edit : DfnEdit::none, DfnImport{edited ? io.addr : 0, pitch, 2});
                auto t2 = c.dev.submit();
                post = c.dev.span();
                if (f >= 2 && !r.from) {
                    r.post_each.resize(t2.size());
                    for (size_t i = 0; i < t2.size(); ++i) r.post_each[i] += t2[i];
                }
            }
            if (f >= 2) r.net_ms += net, r.post_ms += post, ++r.timed;
            c.dev.download(out, po.data(), po.size() * 2);
            FrameHashes h{fnv(po.data(), po.size() * 2), 0, 0, 0};
            for (size_t i = 1; i < po.size(); i += 4) h.green += to_float(po[i]);
            static std::vector<std::vector<uint16_t>> dbg_base;
            if (&r == &runs[0]) dbg_base.push_back(po);
            else if (getenv("ONR_NRDFN_DIFF") && f < dbg_base.size()) {
                size_t n = 0; float mx = 0; size_t first = ~size_t(0);
                for (size_t i = 0; i < po.size(); ++i)
                    if (po[i] != dbg_base[f][i]) { ++n; mx = std::max(mx, std::fabs(to_float(po[i]) - to_float(dbg_base[f][i]))); if (first == ~size_t(0)) first = i; }
                if (n) printf("    %s f%zu: %zu halves differ, max %g, first %zu: %04x vs %04x\n", r.name, f, n, mx, first, first < po.size() ? po[first] : 0, first < po.size() ? dbg_base[f][first] : 0);
            }
            std::vector<uint8_t> hb(e.history().size), db(e.downsampled_history().size);
            c.dev.download(e.history(), hb.data(), hb.size());
            c.dev.download(e.downsampled_history(), db.data(), db.size());
            h.hist = fnv(hb.data(), hb.size()), h.down = fnv(db.data(), db.size());
            r.h.push_back(h);
        }
        e.destroy();
    }

    int bad = 0;
    const Run& base = runs[0];
    printf("  %zu frames, network kernels %s; GPU ms averaged over frames 2..%zu (submit spans)\n", fr.size(),
           fast ? "_f (fast)" : "k8 (faithful)", fr.size() - 1);
    printf("  %-26s %8s %8s %8s  output/history/downsampled vs whole record (frames equal)\n", "run", "net", "post", "total");
    for (const Run& r : runs) {
        int eo = 0, eh = 0, ed = 0;
        for (size_t f = 0; f < r.h.size(); ++f)
            eo += r.h[f].out == base.h[f].out, eh += r.h[f].hist == base.h[f].hist, ed += r.h[f].down == base.h[f].down;
        double n = r.timed ? r.net_ms / r.timed : 0, p = r.timed ? r.post_ms / r.timed : 0;
        const int N = int(r.h.size());
        bool ok;
        if (r.from) ok = eo == r.from && eh == N && ed == N;
        else if (!r.tint) ok = eo == N && eh == N && ed == N;
        else if (r.edit == DfnEdit::display) ok = eo == 0 && eh == N && ed == N;
        else ok = eo == 0 && eh == 0 && ed == 0;
        printf("  %-26s %8.2f %8.2f %8.2f  %d/%d/%d of %d  %s\n", r.name, n, p, n + p, eo, eh, ed, N, ok ? "ok" : "FAIL");
        bad += !ok;
    }
    // every frame's output and histories in one hash per run, to compare builds
    for (const Run& r : runs) printf("  %-26s %016llx\n", r.name, (unsigned long long)fnv(r.h.data(), r.h.size() * sizeof(FrameHashes)));
    // how much of the tint reaches the output: 1 = all of it (G scaled by tint[1]), 0 = none
    printf("  tint reaching the output, frame by frame (1 = all):\n");
    for (const Run& r : runs) {
        if (!r.tint) continue;
        printf("  %-26s", r.name);
        for (size_t f = 0; f < r.h.size(); ++f) printf(" %.2f", (1 - r.h[f].green / base.h[f].green) / (1 - tint[1]));
        printf("\n");
    }
    // the post half dispatch by dispatch (k8; _f has no fixup): none = post, post_fixup, downsample (the tile came from
    // the dec0_1 head); loop = dfn_import, post_tile first; display = dfn_import, post_dlt, post_fixup_dlt, downsample
    printf("  per dispatch (ms): dfn_export | the post half in order\n");
    for (const Run& r : runs) {
        if (!r.split || !r.timed || r.from) continue;
        printf("  %-26s %6.3f |", r.name, r.export_ms / r.timed);
        for (double t : r.post_each) printf(" %6.3f", t / r.timed);
        printf("\n");
    }
    for (auto& b : gb) c.dev.destroy(b);
    c.dev.destroy(out);
    c.dev.destroy(io);
    return bad;
}
