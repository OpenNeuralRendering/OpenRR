// SPDX-License-Identifier: MIT
// GPU harness: runs each rr2 kernel on the captured inputs (case folders written by
// gpu/tools/prep_cases.py) and compares with the captured outputs and with ref-impl.
#include "harness.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

bool CaseFile::load(const fs::path& d) {
    dir = d;
    std::ifstream f(d / "case.txt");
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string key, tok;
        if (!(ss >> key)) continue;
        std::vector<std::string> v;
        while (ss >> tok) v.push_back(tok);
        kv.emplace(key, v);
    }
    return true;
}

const std::vector<std::string>& CaseFile::get(const std::string& k) const {
    auto it = kv.find(k);
    if (it == kv.end()) throw std::runtime_error("case key missing: " + k);
    return it->second;
}

std::vector<std::vector<std::string>> CaseFile::all(const std::string& k) const {
    std::vector<std::vector<std::string>> r;
    for (auto [a, b] = kv.equal_range(k); a != b; ++a) r.push_back(a->second);
    return r;
}

int CaseFile::i(const std::string& k, size_t n) const { return std::stoi(get(k).at(n), nullptr, 0); }
uint64_t CaseFile::u(const std::string& k, size_t n) const { return std::stoull(get(k).at(n), nullptr, 0); }
float CaseFile::f(const std::string& k, size_t n) const { return bitsf(uint32_t(std::stoul(get(k).at(n), nullptr, 16))); }

template <class T> std::vector<T> CaseFile::read(const std::string& name) const {
    std::ifstream f(dir / name, std::ios::binary);
    if (!f) throw std::runtime_error("missing " + (dir / name).string());
    f.seekg(0, std::ios::end);
    size_t n = size_t(f.tellg());
    f.seekg(0);
    std::vector<T> v(n / sizeof(T));
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
    return v;
}
template std::vector<uint8_t> CaseFile::read(const std::string&) const;
template std::vector<uint16_t> CaseFile::read(const std::string&) const;
template std::vector<uint32_t> CaseFile::read(const std::string&) const;
template std::vector<float> CaseFile::read(const std::string&) const;

uint32_t fbits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
float bitsf(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }

float to_float(uint16_t h) {
    uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) return (s ? -1.f : 1.f) * std::ldexp(float(m), -24);
    if (e == 31) return bitsf(s | 0x7f800000u | (m << 13));
    return bitsf(s | ((e + 112) << 23) | (m << 13));
}

double cmp_f16(const char* what, const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, int ch, int use) {
    size_t n = std::min(a.size(), b.size()), cnt = 0, eq = 0, nan = 0;
    double mse = 0, peak = 0, maxd = 0;
    for (size_t i = 0; i < n; ++i) {
        if (ch && use && int(i % ch) >= use) continue;
        ++cnt;
        if (a[i] == b[i]) { ++eq; }
        double x = to_float(a[i]), y = to_float(b[i]);
        if (std::isnan(x) || std::isnan(y)) { nan += std::isnan(x) != std::isnan(y); continue; }
        double d = std::fabs(x - y);
        maxd = std::max(maxd, d);
        mse += d * d;
        peak = std::max(peak, std::fabs(y));
    }
    mse /= double(cnt ? cnt : 1);
    double psnr = mse == 0 ? INFINITY : 10 * std::log10(peak * peak / mse);
    double frac = double(eq) / double(cnt ? cnt : 1);
    printf("  %-44s bit-equal %.6f%% (%zu of %zu differ)  max_abs %.4g  psnr %.2f dB%s\n", what, 100 * frac, cnt - eq, cnt,
           maxd, psnr, nan ? "  NAN MISMATCH" : "");
    return frac;
}

double cmp_1010102(const char* what, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    size_t n = std::min(a.size(), b.size()), eq = 0, words = 0;
    int maxd = 0;
    for (size_t i = 0; i < n; ++i) {
        words += a[i] == b[i];
        for (int s = 0; s < 30; s += 10) {
            int x = int((a[i] >> s) & 1023), y = int((b[i] >> s) & 1023);
            eq += x == y;
            maxd = std::max(maxd, std::abs(x - y));
        }
    }
    double frac = double(eq) / double(3 * n);
    printf("  %-44s codes equal %.6f%% (%zu of %zu differ, max %d code)  words equal %.6f%%\n", what, 100 * frac,
           3 * n - eq, 3 * n, maxd, 100.0 * double(words) / double(n));
    return frac;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    fs::path cases = argc > 1 ? argv[1] : "out/gpu-cases";
    if (cases == "weights" && argc > 3) {   // weights <nvngx_dlssd.dll> <cases>: C++ weight builder against the Python reference
        try { return run_weights(argv[2], argv[3]); } catch (const std::exception& e) { printf("  ERROR %s\n", e.what()); return 1; }
    }
    bool d3d = cases == "d3d12";   // d3d12 <cases> <case...>: Vulkan against D3D12 on the same inputs (run_d3d12)
    if (d3d) cases = argc > 2 ? argv[2] : "out/gpu-cases";
    std::string spv = !d3d && argc > 2 && cases != "probe" && cases != "chain" && cases != "seq" && cases != "optguides" && strcmp(argv[2], "-") ? argv[2] : ONR_SPV_DIR;
    if (const char* s = getenv("ONR_SPV")) spv = s;   // kernel directory override (also for chain / seq)
    std::vector<std::string> only(argv + std::min(argc, 3), argv + argc);
    Ctx c;
    if (const char* r = getenv("ONR_REPS")) c.reps = std::max(1, atoi(r));
    std::string err;
    if (!c.dev.init(err)) { fprintf(stderr, "vulkan: %s\n", err.c_str()); return 2; }
    auto& di = c.dev.info();
    printf("device: %s, driver %s, fp8 coopmat %d, fp64 %d, int64 atomics %d\n", di.name.c_str(), di.driver_info.c_str(),
           di.fp8_coopmat, di.fp64, di.int64_atomics);
    if (cases == "engine" && argc > 3) {   // engine <nvngx_dlssd.dll> <cases>: the runtime's frame graph on the sequence
        int r = 1;
        try { r = run_engine(c, argv[2], argv[3]); } catch (const std::exception& e) { printf("  ERROR %s\n", e.what()); }
        c.k.unload(c.dev);
        return r;
    }
    if (cases == "nrdfn" && argc > 3) {    // nrdfn <nvngx_dlssd.dll> <cases>: the denoised-image hop (FASE5-NR.md)
        int r = 1;
        try { r = run_nrdfn(c, argv[2], argv[3]); } catch (const std::exception& e) { printf("  ERROR %s\n", e.what()); }
        c.k.unload(c.dev);
        return r;
    }
    c.k.load(c.dev, spv);
    if (getenv("ONR_VERBOSE"))
        for (const char* n : {"downsample", "post_tile", "post"}) printf("%s: %s\n", n, c.dev.stats(c.k[n]).c_str());
    if (cases == "chain") {
        int r = 1;
        try { r = run_chain(c, argc > 2 ? fs::path(argv[2]) : fs::path("out/gpu-cases")); } catch (const std::exception& e) { printf("  ERROR %s\n", e.what()); }
        c.k.unload(c.dev);
        return r;
    }
    if (cases == "optguides") {   // optguides <cases>: rr2_enc0's optional guides (docs/NOTA-ENTRADAS.md) on the enc0_f10 case
        int r = 1;
        try { r = run_optguides(c, argc > 2 ? fs::path(argv[2]) : fs::path("out/gpu-cases")); } catch (const std::exception& e) { printf("  ERROR %s\n", e.what()); }
        c.k.unload(c.dev);
        return r;
    }
    if (cases == "seq") {
        int r = 1;
        try { r = run_seq(c, argc > 2 ? fs::path(argv[2]) : fs::path("out/gpu-cases")); } catch (const std::exception& e) { printf("  ERROR %s\n", e.what()); }
        c.k.unload(c.dev);
        return r;
    }
    if (cases == "probe") { int r = run_probes(c, argc > 2 ? fs::path(argv[2]) : fs::path("out/gpu-cases")); c.k.unload(c.dev); return r; }
    int fails = 0;
    std::vector<fs::path> dirs;
    for (auto& e : fs::directory_iterator(cases))
        if (e.is_directory()) dirs.push_back(e.path());
    std::sort(dirs.begin(), dirs.end());
    for (auto& d : dirs) {
        std::string n = d.filename().string();
        if (!only.empty() && std::none_of(only.begin(), only.end(), [&](auto& o) { return n.rfind(o, 0) == 0; })) continue;
        CaseFile cf;
        if (!cf.load(d)) continue;
        printf("== %s\n", n.c_str());
        try {
            if (d3d) fails += run_d3d12(c, cf);
            else if (n.rfind("clear", 0) == 0) fails += run_clear(c, cf);
            else if (n.rfind("pooled", 0) == 0) fails += run_pooled(c, cf);
            else if (n.rfind("ae_", 0) == 0) fails += run_ae(c, cf);
            else if (n.rfind("ds_", 0) == 0) fails += run_ds(c, cf);
            else if (n.rfind("post", 0) == 0) fails += run_post(c, cf);
            else if (n.rfind("enc", 0) == 0 || n.rfind("dec", 0) == 0 || n.rfind("blk", 0) == 0) fails += run_enc(c, cf);
        } catch (const std::exception& e) {
            printf("  ERROR %s\n", e.what());
            ++fails;
        }
    }
    c.k.unload(c.dev);
    printf(fails ? "%d case(s) below their bar\n" : "all cases pass (bit-exact, post within tolerance)\n", fails);
    return fails ? 1 : 0;
}
