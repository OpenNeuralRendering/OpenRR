// SPDX-License-Identifier: MIT
#pragma once
#include "opennr_rr/kernels.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct CaseFile {
    fs::path dir;
    std::multimap<std::string, std::vector<std::string>> kv;
    bool load(const fs::path& d);
    const std::vector<std::string>& get(const std::string& k) const;
    std::vector<std::vector<std::string>> all(const std::string& k) const;
    int i(const std::string& k, size_t n = 0) const;
    float f(const std::string& k, size_t n = 0) const;           // hex bits
    uint64_t u(const std::string& k, size_t n = 0) const;
    template <class T> std::vector<T> read(const std::string& name) const;
    bool has_file(const std::string& name) const { return fs::exists(dir / name); }
};

struct Ctx {
    onr::Device dev;
    onr::Kernels k;
    int reps = 20;
};

float to_float(uint16_t h);
uint32_t fbits(float f);
float bitsf(uint32_t b);

// metrics, printed as one line; returns the fraction of equal elements
double cmp_f16(const char* what, const std::vector<uint16_t>& ours, const std::vector<uint16_t>& ref, int ch = 0, int use = 0);
double cmp_1010102(const char* what, const std::vector<uint32_t>& ours, const std::vector<uint32_t>& ref);
// median GPU time (ms) of `reps` recordings of fn (the sum of all timed dispatches of a recording)
template <class F> double time_it(Ctx& c, F&& fn);

int run_clear(Ctx&, const CaseFile&);
int run_pooled(Ctx&, const CaseFile&);
int run_ae(Ctx&, const CaseFile&);
int run_ds(Ctx&, const CaseFile&);
int run_post(Ctx&, const CaseFile&);
int run_enc(Ctx&, const CaseFile&);
int run_net(Ctx&, const CaseFile&);
int run_enc0(Ctx&, const CaseFile&);            // rr2_enc0 (net_case.cpp)             // enc1, decoders, dec0_1 head (net_case.cpp)
int run_d3d12(Ctx&, const CaseFile&);          // the case's kernel on Vulkan and on D3D12 (net_case.cpp, FASE9-D3D12)
int run_chain(Ctx&, const fs::path& cases);     // whole frame 10 (own enc0, histories from the capture)
int run_seq(Ctx&, const fs::path& cases);       // frames 0..N of the sequence, every state our own
int run_optguides(Ctx&, const fs::path& cases);   // rr2_enc0 optional guides on enc0_f10 (net_case.cpp)
double cmp_e4m3(const char* what, const std::vector<uint8_t>& ours, const std::vector<uint8_t>& ref);
onr::EncBlockPC load_block(Ctx&, const CaseFile&, std::vector<onr::Buffer>& keep, uint64_t x, uint64_t y);
int run_probes(Ctx&);
int run_weights(const fs::path& dll, const fs::path& cases);   // weights_case.cpp
int run_engine(Ctx&, const fs::path& dll, const fs::path& cases);   // engine_case.cpp
int run_nrdfn(Ctx&, const fs::path& dll, const fs::path& cases);    // nrdfn_case.cpp
int run_probes(Ctx&, const fs::path& cases);

#include <algorithm>
#include <cstdio>
#include <cstdlib>
template <class F> double time_it(Ctx& c, F&& fn) {
    std::vector<double> t;
    for (int r = 0; r < c.reps; ++r) {
        c.dev.begin();
        fn();
        auto ms = c.dev.submit();
        double s = 0;
        for (double x : ms) s += x;
        if (r == c.reps - 1 && getenv("ONR_VERBOSE")) { for (double x : ms) printf(" %.4f", x); printf(" ms per dispatch, span %.4f ms\n", c.dev.span()); }
        t.push_back(s);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}
