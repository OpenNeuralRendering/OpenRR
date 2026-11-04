// SPDX-License-Identifier: MIT
// Drives OpenNrRrRuntime.dll through its D3D12 path as a game would: the scene's guides as D3D12 textures,
// RecordInputs on one list, Enqueue between two ExecuteCommandLists, RecordOutput on the second, and compares each
// frame's output texture with the output NGX produced for the same frame in a capture.
//
// usage: opennr_rr_validate <runtime.dll> <data dir> <nvngx_dlssd.dll> <scene dir> <capture frames dir> <out WxH>
//                           [frames] [fast|faithful] [split|loop|display]
// split, loop, display: ABI 2, EnqueueNetwork + EnqueuePost; loop and display send the denoised image through a
// texture of the host and back (no edit), so the output must match the plain run bit for bit.
// ONR_VALIDATE_HALF=1: the image asked for as rgba16f. ONR_VALIDATE_COPIES=n: n buffer -> texture -> buffer round trips (0: none)
// in the hop (2 = the four copies of the mochizuki hop). With ONR_VALIDATE_LOOP, the hop's timeline on the GPU (ABI 3
// GetHopTiming and D3D12 timestamps around the host's lists, in QueryPerformanceCounter ticks).
// ONR_VALIDATE_BACKEND=auto|d3d12|vulkan: the backend asked for at Create (default vulkan). With the D3D12 backend
// (ABI 4) the frame is one Record on the game's list (a hop: Record, the host's list, RecordPost on the second list).
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <cmath>
#include <cstdio>
#include <cctype>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "opennr_rr_api.h"

namespace {

void hr_check(HRESULT hr, const char* what) {
    if (FAILED(hr)) throw std::runtime_error(std::string(what) + " failed");
}

std::vector<uint8_t> read_file(const std::wstring& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read a scene file");
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
}

// numbers of key[...] in the n-th frame object of sequence.json (the frames array is the last thing with these keys)
std::vector<double> json_list(const std::string& js, size_t from, const char* key) {
    size_t k = js.find(std::string("\"") + key + "\"", from);
    std::vector<double> v;
    if (k == std::string::npos) return v;
    size_t p = js.find_first_of("[0123456789-", js.find(':', k) + 1);
    if (js[p] != '[') return {strtod(js.c_str() + p, nullptr)};
    ++p;
    while (isspace(uint8_t(js[p]))) ++p;
    while (js[p] != ']') {
        char* e;
        v.push_back(strtod(js.c_str() + p, &e));
        p = size_t(e - js.c_str());
        while (js[p] == ',' || isspace(uint8_t(js[p]))) ++p;
    }
    return v;
}

float h2f(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp(float(m), -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(float(m | 1024), int(e) - 25);
    return s ? -v : v;
}

struct Tex {
    ID3D12Resource* res = nullptr;
    DXGI_FORMAT fmt;
    UINT w, h, bpp;
};

// ONR_VALIDATE_RE9: the scene in Resident Evil Requiem's formats, written and read here on the CPU on their own
// (independently of the runtime's kernels)
uint16_t f2h_exact(double x) {   // x exactly representable in fp16, >= 0
    if (x == 0) return 0;
    int k = 0;
    std::frexp(x, &k);
    if (k - 1 < -14) return uint16_t(std::ldexp(x, 24));
    return uint16_t((k + 14) << 10 | uint32_t(std::ldexp(x, -(k - 1)) * 1024 - 1024));
}
// R9G9B9E5 with round-to-zero (the D3D rule, as the GPU's own typed store does)
uint32_t rgb9e5_of(const float c[3]) {
    double v[3], mx = 0;
    for (int i = 0; i < 3; ++i) v[i] = c[i] > 0 ? std::min(double(c[i]), 65408.0) : 0.0, mx = std::max(mx, v[i]);
    const int e = mx >= std::ldexp(1.0, -16) ? int(std::floor(std::log2(mx))) + 16 : 0;
    uint32_t r = uint32_t(e) << 27;
    for (int i = 0; i < 3; ++i) r |= uint32_t(std::floor(v[i] / std::ldexp(1.0, e - 24))) << (9 * i);
    return r;
}
double rgb9e5_ch(uint32_t u, int i) { return std::ldexp(double((u >> (9 * i)) & 511), int(u >> 27) - 24); }
// unsigned small float, n mantissa bits, bias 15, truncated (values in [0, 1] here)
uint32_t uf_of(double x, int n) {
    if (x < std::ldexp(1.0, -14)) return uint32_t(std::ldexp(x, 14 + n));
    const int k = int(std::floor(std::log2(x)));
    return uint32_t(k + 15) << n | uint32_t(std::ldexp(std::ldexp(x, -k) - 1, n));
}
double uf_val(uint32_t v, int n) {
    const uint32_t e = v >> n, m = v & ((1u << n) - 1);
    return e ? std::ldexp(double(m | 1u << n), int(e) - 15 - n) : std::ldexp(double(m), -14 - n);
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    if (argc < 7) {
        printf("usage: opennr_rr_validate <runtime.dll> <data dir> <nvngx_dlssd.dll> <scene dir> <capture frames dir> <WxH> [frames] [fast|faithful] "
               "[split|loop|display]\n");
        return 2;
    }
    try {
        UINT OW = 0, OH = 0;
        swscanf_s(argv[6], L"%ux%u", &OW, &OH);
        int nframes = argc > 7 ? _wtoi(argv[7]) : 12;
        bool faithful = argc > 8 && !wcscmp(argv[8], L"faithful");
        // -1: ABI 1 Enqueue; otherwise the OnrRrEdit given to EnqueueNetwork
        int edit = -1;
        if (argc > 9) edit = !wcscmp(argv[9], L"loop") ? ONR_RR_EDIT_LOOP : !wcscmp(argv[9], L"display") ? ONR_RR_EDIT_DISPLAY : ONR_RR_EDIT_OFF;
        std::wstring scene = argv[4], frames = argv[5];
        std::string js;
        {
            auto b = read_file(scene + L"\\sequence.json");
            js.assign(b.begin(), b.end());
        }
        auto rs = json_list(js, 0, "render_size");
        UINT RW = UINT(rs[0]), RH = UINT(rs[1]);
        size_t fr0 = js.find("\"frames\"");

        HMODULE dll = LoadLibraryW(argv[1]);
        if (!dll) throw std::runtime_error("cannot load the runtime DLL");
        auto get = reinterpret_cast<PFN_OnrRrGetApi>(GetProcAddress(dll, "OnrRrGetApi"));
        OnrRrApi4 a4{};
        OnrRrApi3& a3 = a4.v3;
        OnrRrApi2& a2 = a3.v2;
        if (!get) throw std::runtime_error("OnrRrGetApi");
        if (get(ONR_RR_ABI_VERSION_4, &a2.v1)) {
            a4 = {};
            if (get(ONR_RR_ABI_VERSION_3, &a2.v1)) throw std::runtime_error("OnrRrGetApi");
        }
        OnrRrApi& api = a2.v1;
        const bool half = getenv("ONR_VALIDATE_HALF") != nullptr;
        const int copies = getenv("ONR_VALIDATE_COPIES") ? std::max(0, atoi(getenv("ONR_VALIDATE_COPIES"))) : 1;

        IDXGIFactory6* fac = nullptr;
        hr_check(CreateDXGIFactory2(0, IID_PPV_ARGS(&fac)), "CreateDXGIFactory2");
        IDXGIAdapter1* ad = nullptr;
        hr_check(fac->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&ad)), "adapter");
        DXGI_ADAPTER_DESC1 add;
        ad->GetDesc1(&add);
        ID3D12Device* dev = nullptr;
        hr_check(D3D12CreateDevice(ad, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)), "D3D12CreateDevice");
        printf("D3D12 adapter: %ls\n", add.Description);
        LUID luid = dev->GetAdapterLuid();
        char err[512] = {};
        if (api.Check(&luid, argv[3])) {
            api.GetLastError(err, sizeof err);
            printf("Check: unsupported: %s\n", err);
            return 1;
        }
        D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
        ID3D12CommandQueue* q = nullptr;
        hr_check(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q)), "queue");
        // 0, 1: the game's halves; 2: the host's hop; 3, 4: a timestamp before EnqueueNetwork and after EnqueuePost
        ID3D12CommandAllocator* al[5];
        ID3D12GraphicsCommandList* cl[5];
        for (int i = 0; i < 5; ++i) {
            hr_check(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&al[i])), "allocator");
            hr_check(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, al[i], nullptr, IID_PPV_ARGS(&cl[i])), "list");
            cl[i]->Close();
        }
        ID3D12Fence* fence = nullptr;
        hr_check(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "fence");
        uint64_t fv = 0;
        HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        auto wait_gpu = [&] {
            q->Signal(fence, ++fv);
            if (fence->GetCompletedValue() < fv) fence->SetEventOnCompletion(fv, ev), WaitForSingleObject(ev, INFINITE);
        };

        auto make_tex = [&](UINT w, UINT h, DXGI_FORMAT f, UINT bpp, D3D12_RESOURCE_FLAGS fl, D3D12_RESOURCE_STATES st) {
            D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
            D3D12_RESOURCE_DESC rd{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, w, h, 1, 1, f, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, fl};
            Tex t{nullptr, f, w, h, bpp};
            hr_check(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, st, nullptr, IID_PPV_ARGS(&t.res)), "texture");
            return t;
        };
        auto make_buf = [&](UINT64 n, D3D12_HEAP_TYPE type) {
            D3D12_HEAP_PROPERTIES hp{type};
            D3D12_RESOURCE_DESC rd{D3D12_RESOURCE_DIMENSION_BUFFER, 0, n, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
            ID3D12Resource* r = nullptr;
            hr_check(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                  type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST,
                                                  nullptr, IID_PPV_ARGS(&r)),
                     "buffer");
            return r;
        };
        const D3D12_RESOURCE_STATES SRV = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        // ONR_VALIDATE_RE9=1: the guides in Resident Evil Requiem's formats (colour R9G9B9E5, albedos R11G11B10, depth in
        // the depth plane of an R32G8X24 depth-stencil, hit distance R32, roughness in normals.w and no roughness
        // texture) and an R9G9B9E5 output; =ref: the same values in the formats above (rgba16f colour / albedos with
        // alpha 1, R32 depth) and an rgba16f output whose hash is taken over its R9G9B9E5 encoding: both must give the
        // same hashes
        const char* re9e = getenv("ONR_VALIDATE_RE9");
        const int re9 = !re9e ? 0 : !strcmp(re9e, "ref") ? 2 : 1;
        struct G { const wchar_t* dir; DXGI_FORMAT f; UINT bpp; DXGI_FORMAT up; D3D12_RESOURCE_FLAGS fl; };
        G gs[8] = {{L"color", DXGI_FORMAT_R16G16B16A16_FLOAT, 8}, {L"diffuse_albedo", DXGI_FORMAT_R8G8B8A8_UNORM, 4},
                   {L"specular_albedo", DXGI_FORMAT_R8G8B8A8_UNORM, 4}, {L"normals", DXGI_FORMAT_R16G16B16A16_FLOAT, 8},
                   {L"roughness", DXGI_FORMAT_R16_FLOAT, 2}, {L"depth", DXGI_FORMAT_R32_FLOAT, 4},
                   {L"motion_vectors", DXGI_FORMAT_R16G16_FLOAT, 4}, {L"specular_hit_distance", DXGI_FORMAT_R16_FLOAT, 2}};
        if (re9 == 1) {
            gs[0].f = DXGI_FORMAT_R9G9B9E5_SHAREDEXP, gs[0].bpp = 4;
            gs[1].f = gs[2].f = DXGI_FORMAT_R11G11B10_FLOAT;
            gs[5].f = DXGI_FORMAT_R32G8X24_TYPELESS, gs[5].up = DXGI_FORMAT_R32_TYPELESS, gs[5].fl = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
            gs[7].f = DXGI_FORMAT_R32_FLOAT, gs[7].bpp = 4;
        } else if (re9 == 2) {
            gs[1].f = gs[2].f = DXGI_FORMAT_R16G16B16A16_FLOAT, gs[1].bpp = gs[2].bpp = 8;
        }
        for (auto& g : gs)
            if (!g.up) g.up = g.f;
        // RE9's output: R9G9B9E5 with a UAV when the GPU takes one (the game's has render target and UAV flags)
        D3D12_FEATURE_DATA_FORMAT_SUPPORT e5{DXGI_FORMAT_R9G9B9E5_SHAREDEXP};
        const bool e5_uav = SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &e5, sizeof e5)) &&
                            (e5.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW);
        const D3D12_RESOURCE_STATES out_state = re9 == 1 && !e5_uav ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        Tex tex[8];
        for (int i = 0; i < 8; ++i) tex[i] = make_tex(RW, RH, gs[i].f, gs[i].bpp, gs[i].fl, SRV);
        Tex out = re9 == 1 ? make_tex(OW, OH, DXGI_FORMAT_R9G9B9E5_SHAREDEXP, 4, e5_uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE, out_state)
                           : make_tex(OW, OH, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (re9) {
            D3D12_RESOURCE_DESC dd = tex[5].res->GetDesc();
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
            dev->GetCopyableFootprints(&dd, 0, 1, 0, &fp, nullptr, nullptr, nullptr);
            printf("RE9 formats (%s): colour %d, albedos %d, depth %d (plane 0 footprint: format %d, row pitch %u), hit %d, output %d%s\n",
                   re9 == 1 ? "as the game" : "reference", int(gs[0].f), int(gs[1].f), int(gs[5].f), int(fp.Footprint.Format), fp.Footprint.RowPitch,
                   int(gs[7].f), int(out.fmt), re9 == 1 ? (e5_uav ? " with a UAV" : " without a UAV") : "");
        }
        // ONR_VALIDATE_DM_SHIFT=n (with ONR_VALIDATE_DM): the mask is a texture of its own, roughness * 2^n (exact in
        // fp16, values far above 1 as Resident Evil Requiem's); ONR_VALIDATE_DM_PRECLAMP=1 clamps it to 0..1 here, so
        // the runtime's ONR_RR_FRAME_DM_CLAMP (ONR_VALIDATE_DM_CLAMP=1) on the unclamped mask must give the same hashes
        const int dm_shift = getenv("ONR_VALIDATE_DM_SHIFT") ? atoi(getenv("ONR_VALIDATE_DM_SHIFT")) : 0;
        const bool dm_preclamp = getenv("ONR_VALIDATE_DM_PRECLAMP") != nullptr;
        Tex dmt = dm_shift ? make_tex(RW, RH, DXGI_FORMAT_R16_FLOAT, 2, D3D12_RESOURCE_FLAG_NONE, SRV) : Tex{};
        // ONR_VALIDATE_TL_SPARSE=p (measurements): the layer is the colour in 32x32 tiles covering ~p% of the frame and +0
        // elsewhere (particles over a mostly empty layer)
        const int tl_sparse = getenv("ONR_VALIDATE_TL_SPARSE") ? atoi(getenv("ONR_VALIDATE_TL_SPARSE")) : -1;
        Tex tlt = tl_sparse >= 0 ? make_tex(RW, RH, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, D3D12_RESOURCE_FLAG_NONE, SRV) : Tex{};
        // ONR_VALIDATE_TLO=1 (with ONR_VALIDATE_TL): the layer's alpha as its opacity, in every channel; the opacity's path
        // must give the layer-only hashes. ONR_VALIDATE_RESP=0 / ONR_VALIDATE_DOF=0: a zero texture as the mask / guide
        // (the 64-byte record path must give the hashes without them). ONR_VALIDATE_TLMV=1: zero layer motion vectors at
        // the output size (committed resources start zeroed)
        const char* tloe = getenv("ONR_VALIDATE_TLO");
        Tex tot = tloe && !strcmp(tloe, "1") ? make_tex(RW, RH, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, D3D12_RESOURCE_FLAG_NONE, SRV) : Tex{};
        Tex zt = make_tex(RW, RH, DXGI_FORMAT_R16_FLOAT, 2, D3D12_RESOURCE_FLAG_NONE, SRV);
        Tex tmv = getenv("ONR_VALIDATE_TLMV") ? make_tex(OW, OH, DXGI_FORMAT_R16G16_FLOAT, 4, D3D12_RESOURCE_FLAG_NONE, SRV) : Tex{};
        UINT64 up_stride = ((UINT64(RW) * 8 + 255) & ~255ull) * RH;
        ID3D12Resource* upload = make_buf(up_stride * 11, D3D12_HEAP_TYPE_UPLOAD);
        UINT opitch = OW * out.bpp;
        ID3D12Resource* rb = make_buf(UINT64(opitch) * OH, D3D12_HEAP_TYPE_READBACK);
        Tex hop = make_tex(RW, RH, half ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R32G32B32A32_FLOAT, half ? 8 : 16, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_COMMON);
        // timestamps: before EnqueueNetwork, hop list start, hop list end, after EnqueuePost
        D3D12_QUERY_HEAP_DESC qhd{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 4, 0};
        ID3D12QueryHeap* qh = nullptr;
        hr_check(dev->CreateQueryHeap(&qhd, IID_PPV_ARGS(&qh)), "query heap");
        ID3D12Resource* qrb = make_buf(4 * 8, D3D12_HEAP_TYPE_READBACK);
        UINT64 tsf = 0;
        q->GetTimestampFrequency(&tsf);
        LARGE_INTEGER qpf;
        QueryPerformanceFrequency(&qpf);
        struct HostTimes { uint64_t pre, h0, h1, post; };
        std::map<uint64_t, HostTimes> host_times;   // by hop number
        uint64_t hops = 0;
        // ONR_VALIDATE_LOAD=n: n copies of 256 MB on the game's list before the evaluate (a game's frame, so the clock and
        // the queue look like in play); the path's timeline then comes from timestamps around RecordInputs / RecordOutput
        // and, with ONR_RR_TIME_ALL in the runtime, its Vulkan span (GetHopTiming numbers every network frame)
        const int load = getenv("ONR_VALIDATE_LOAD") ? std::max(0, atoi(getenv("ONR_VALIDATE_LOAD"))) : 0;
        ID3D12Resource* lsrc = load ? make_buf(256ull << 20, D3D12_HEAP_TYPE_DEFAULT) : nullptr;
        ID3D12Resource* ldst = load ? make_buf(256ull << 20, D3D12_HEAP_TYPE_DEFAULT) : nullptr;
        ID3D12QueryHeap* ph = nullptr;
        D3D12_QUERY_HEAP_DESC phd{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 4, 0};
        hr_check(dev->CreateQueryHeap(&phd, IID_PPV_ARGS(&ph)), "query heap");
        ID3D12Resource* prb = make_buf(4 * 8, D3D12_HEAP_TYPE_READBACK);
        std::map<uint64_t, HostTimes> path_times;   // by network frame: inputs start / end, output start / end
        double enqueue_cpu = 0;
        int enqueue_n = 0;

        const char* be = getenv("ONR_VALIDATE_BACKEND");
        const uint32_t choice = !be ? ONR_RR_BACKEND_CHOICE_VULKAN : !_stricmp(be, "d3d12") ? ONR_RR_BACKEND_CHOICE_D3D12
                                : !_stricmp(be, "auto")                       ? ONR_RR_BACKEND_CHOICE_AUTO
                                                                              : ONR_RR_BACKEND_CHOICE_VULKAN;
        OnrRrCreateInfo ci{sizeof ci, faithful ? uint32_t(ONR_RR_MODE_FAITHFUL) : uint32_t(ONR_RR_MODE_FAST), dev, RW, RH, OW, OH, argv[3],
                           nullptr, argv[2], nullptr, choice};
        void* ctx = nullptr;
        if (api.Create(&ci, &ctx)) {
            api.GetLastError(err, sizeof err);
            throw std::runtime_error(std::string("Create: ") + err);
        }
        // ONR_VALIDATE_IDLE=1: a second context that never records, so two networks live in the process
        void* idle = nullptr;
        if (getenv("ONR_VALIDATE_IDLE")) {
            OnrRrCreateInfo ci2 = ci;
            ci2.max_render_width = 426, ci2.max_render_height = 240, ci2.output_width = 640, ci2.output_height = 360;
            if (api.Create(&ci2, &idle)) throw std::runtime_error("Create (idle)");
        }
        uint32_t backend = ONR_RR_BACKEND_VULKAN;
        if (a4.GetBackend) a4.GetBackend(ctx, &backend);
        printf("render %ux%u -> %ux%u, %d frames, %s, backend %s\n", RW, RH, OW, OH, nframes, faithful ? "faithful" : "fast",
               backend == ONR_RR_BACKEND_D3D12 ? "D3D12" : "Vulkan");

        // the host's hop, recorded: the image buffer -> texture -> buffer on the same queue (cl[2], closed)
        auto hop_lists = [&](const OnrRrDenoisedImage& im) {
            auto* buf = static_cast<ID3D12Resource*>(im.buffer);
            al[2]->Reset();
            cl[2]->Reset(al[2], nullptr);
            auto tr = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b2) {
                D3D12_RESOURCE_BARRIER t{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION};
                t.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, a, b2};
                cl[2]->ResourceBarrier(1, &t);
            };
            D3D12_TEXTURE_COPY_LOCATION bl{buf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
            bl.PlacedFootprint = {0, {DXGI_FORMAT(im.format), im.width, im.height, 1, im.row_pitch}};
            D3D12_TEXTURE_COPY_LOCATION tl{hop.res, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
            cl[2]->EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            for (int n = 0; n < copies; ++n) {
                tr(buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
                tr(hop.res, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
                cl[2]->CopyTextureRegion(&tl, 0, 0, 0, &bl, nullptr);
                tr(hop.res, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
                tr(buf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
                cl[2]->CopyTextureRegion(&bl, 0, 0, 0, &tl, nullptr);
                tr(buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
                tr(hop.res, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
            }
            cl[2]->EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, 2);
            cl[2]->Close();
        };
        // the output texture into the readback buffer at the end of cl[1], which it closes
        auto readback_output = [&] {
            D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION};
            b.Transition = {out.res, 0, out_state, D3D12_RESOURCE_STATE_COPY_SOURCE};
            cl[1]->ResourceBarrier(1, &b);
            D3D12_TEXTURE_COPY_LOCATION dst{rb, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
            dst.PlacedFootprint = {0, {out.fmt, OW, OH, 1, opitch}};
            D3D12_TEXTURE_COPY_LOCATION src{out.res, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
            cl[1]->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
            cl[1]->ResourceBarrier(1, &b);
            cl[1]->Close();
        };
        bool d3d_frames = false;
        // ONR_VALIDATE_SHARPEN=k (D3D12 backend, a hop in rgba32f): the hop's image edited as x + k (x - box3(x)) on the
        // CPU, a stand-in for an NR that follows the jitter (the display mode's stability on a static scene)
        const float sharpen = getenv("ONR_VALIDATE_SHARPEN") ? float(atof(getenv("ONR_VALIDATE_SHARPEN"))) : 0.0f;
        ID3D12Resource *erb = nullptr, *eup = nullptr;

        // guide i of frame f (the scene's rows, its formats) into gs[i]'s rows
        auto to_re9 = [&](int i, const std::vector<uint8_t>& d, int f) {
            const size_t n = size_t(RW) * RH;
            std::vector<uint8_t> o(n * gs[i].bpp);
            auto h = reinterpret_cast<const uint16_t*>(d.data());
            auto o16 = reinterpret_cast<uint16_t*>(o.data());
            auto o32 = reinterpret_cast<uint32_t*>(o.data());
            switch (i) {
            case 0:   // R9G9B9E5 of the colour; the reference gets its values back, alpha 1
                for (size_t k = 0; k < n; ++k) {
                    const float c[3] = {h2f(h[4 * k]), h2f(h[4 * k + 1]), h2f(h[4 * k + 2])};
                    const uint32_t e = rgb9e5_of(c);
                    if (re9 == 1) o32[k] = e;
                    else for (int ch = 0; ch < 4; ++ch) o16[4 * k + ch] = ch < 3 ? f2h_exact(rgb9e5_ch(e, ch)) : 0x3c00;
                }
                return o;
            case 1: case 2:   // R11G11B10 of the albedo bytes; the reference: its values, alpha 1
                for (size_t k = 0; k < n; ++k) {
                    uint32_t v[3];
                    for (int ch = 0; ch < 3; ++ch) v[ch] = uf_of(d[4 * k + ch] / 255.0, ch < 2 ? 6 : 5);
                    if (re9 == 1) o32[k] = v[0] | v[1] << 11 | v[2] << 22;
                    else for (int ch = 0; ch < 4; ++ch) o16[4 * k + ch] = ch < 3 ? f2h_exact(uf_val(v[ch], ch < 2 ? 6 : 5)) : 0x3c00;
                }
                return o;
            case 3: {   // roughness in normals.w
                wchar_t p[MAX_PATH];
                swprintf_s(p, L"%s\\roughness\\%04d.raw", scene.c_str(), f);
                auto r = read_file(p);
                o = d;
                for (size_t k = 0; k < n; ++k) memcpy(&o[8 * k + 6], &r[2 * k], 2);
                return o;
            }
            case 7:
                if (re9 == 1)
                    for (size_t k = 0; k < n; ++k) {
                        const float v = h2f(h[k]);
                        memcpy(&o32[k], &v, 4);
                    }
                else o = d;
                return o;
            default: return d;
            }
        };
        uint8_t* um = nullptr;        upload->Map(0, nullptr, reinterpret_cast<void**>(&um));
        auto run_frame = [&](int f, bool upload_guides = true) {
            // guides into the textures
            al[0]->Reset();
            cl[0]->Reset(al[0], nullptr);
            for (int i = 0; i < 8 && upload_guides; ++i) {
                wchar_t p[MAX_PATH];
                swprintf_s(p, L"%s\\%s\\%04d.raw", scene.c_str(), gs[i].dir, f);
                auto d = read_file(p);
                if (re9) d = to_re9(i, d, f);
                UINT pitch = (RW * gs[i].bpp + 255) & ~255u;
                for (UINT y = 0; y < RH; ++y) memcpy(um + up_stride * i + UINT64(y) * pitch, d.data() + size_t(y) * RW * gs[i].bpp, size_t(RW) * gs[i].bpp);
                D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION};
                b.Transition = {tex[i].res, 0, SRV, D3D12_RESOURCE_STATE_COPY_DEST};
                cl[0]->ResourceBarrier(1, &b);
                D3D12_TEXTURE_COPY_LOCATION dst{tex[i].res, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
                D3D12_TEXTURE_COPY_LOCATION src{upload, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
                src.PlacedFootprint = {up_stride * i, {gs[i].up, RW, RH, 1, pitch}};
                cl[0]->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
                cl[0]->ResourceBarrier(1, &b);
            }
            if (tl_sparse >= 0 && upload_guides) {
                wchar_t p[MAX_PATH];
                swprintf_s(p, L"%s\\color\\%04d.raw", scene.c_str(), f);
                auto d = read_file(p);
                size_t empty = 0;
                for (UINT y = 0; y < RH; ++y)
                    for (UINT x = 0; x < RW; ++x) {
                        const uint32_t k = ((x >> 5) * 73856093u) ^ ((y >> 5) * 19349663u);
                        if ((k * 2654435761u >> 16) % 100 >= uint32_t(tl_sparse)) memset(d.data() + (size_t(y) * RW + x) * 8, 0, 8);
                    }
                auto zero = [&](int x, int y) {
                    x = std::clamp(x, 0, int(RW) - 1), y = std::clamp(y, 0, int(RH) - 1);
                    const uint64_t* v = reinterpret_cast<const uint64_t*>(d.data()) + size_t(y) * RW + x;
                    return *v == 0;
                };
                for (int y = 0; y < int(RH); ++y)
                    for (int x = 0; x < int(RW); ++x) {
                        bool z = true;
                        for (int dy = -1; dy <= 1 && z; ++dy)
                            for (int dx = -1; dx <= 1 && z; ++dx) z = zero(x + dx, y + dy);
                        empty += z;
                    }
                if (f == 0) printf("sparse layer: %.1f%% of the texels with a +0 3x3\n", 100.0 * double(empty) / (double(RW) * RH));
                const UINT pitch = (RW * 8 + 255) & ~255u;
                for (UINT y = 0; y < RH; ++y) memcpy(um + up_stride * 9 + UINT64(y) * pitch, d.data() + size_t(y) * RW * 8, size_t(RW) * 8);
                D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION};
                b.Transition = {tlt.res, 0, SRV, D3D12_RESOURCE_STATE_COPY_DEST};
                cl[0]->ResourceBarrier(1, &b);
                D3D12_TEXTURE_COPY_LOCATION dst{tlt.res, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
                D3D12_TEXTURE_COPY_LOCATION src{upload, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
                src.PlacedFootprint = {up_stride * 9, {DXGI_FORMAT_R16G16B16A16_FLOAT, RW, RH, 1, pitch}};
                cl[0]->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
                cl[0]->ResourceBarrier(1, &b);
            }
            if (tot.res && upload_guides) {
                wchar_t p[MAX_PATH];
                swprintf_s(p, L"%s\\color\\%04d.raw", scene.c_str(), f);
                auto d = read_file(p);
                auto h = reinterpret_cast<uint16_t*>(d.data());
                for (size_t k = 0; k < size_t(RW) * RH; ++k) h[4 * k] = h[4 * k + 1] = h[4 * k + 2] = h[4 * k + 3];
                const UINT pitch = (RW * 8 + 255) & ~255u;
                for (UINT y = 0; y < RH; ++y) memcpy(um + up_stride * 10 + UINT64(y) * pitch, d.data() + size_t(y) * RW * 8, size_t(RW) * 8);
                D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION};
                b.Transition = {tot.res, 0, SRV, D3D12_RESOURCE_STATE_COPY_DEST};
                cl[0]->ResourceBarrier(1, &b);
                D3D12_TEXTURE_COPY_LOCATION dst{tot.res, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
                D3D12_TEXTURE_COPY_LOCATION src{upload, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
                src.PlacedFootprint = {up_stride * 10, {DXGI_FORMAT_R16G16B16A16_FLOAT, RW, RH, 1, pitch}};
                cl[0]->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
                cl[0]->ResourceBarrier(1, &b);
            }
            if (dm_shift && upload_guides) {
                wchar_t p[MAX_PATH];
                swprintf_s(p, L"%s\\roughness\\%04d.raw", scene.c_str(), f);
                auto r = read_file(p);
                auto h = reinterpret_cast<uint16_t*>(r.data());
                for (size_t k = 0; k < size_t(RW) * RH; ++k) {
                    const double v = std::ldexp(double(h2f(h[k])), dm_shift);
                    h[k] = f2h_exact(dm_preclamp ? std::min(v, 1.0) : v);
                }
                const UINT pitch = (RW * 2 + 255) & ~255u;
                for (UINT y = 0; y < RH; ++y) memcpy(um + up_stride * 8 + UINT64(y) * pitch, r.data() + size_t(y) * RW * 2, size_t(RW) * 2);
                D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION};
                b.Transition = {dmt.res, 0, SRV, D3D12_RESOURCE_STATE_COPY_DEST};
                cl[0]->ResourceBarrier(1, &b);
                D3D12_TEXTURE_COPY_LOCATION dst{dmt.res, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
                D3D12_TEXTURE_COPY_LOCATION src{upload, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
                src.PlacedFootprint = {up_stride * 8, {DXGI_FORMAT_R16_FLOAT, RW, RH, 1, pitch}};
                cl[0]->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
                cl[0]->ResourceBarrier(1, &b);
            }
            // the game's evaluate: first half, between, second half
            size_t at = fr0;
            for (int k = 0; k <= f; ++k) at = js.find("\"jitter\"", at + 1);
            OnrRrFrame fr{};
            fr.size = sizeof fr;
            auto jit = json_list(js, at, "jitter"), mvs = json_list(js, at, "mv_scale"), rst = json_list(js, at, "reset");
            auto w2v = json_list(js, at, "world_to_view"), v2c = json_list(js, at, "view_to_clip");
            fr.render_width = RW, fr.render_height = RH;
            fr.jitter_x = float(jit[0]), fr.jitter_y = float(jit[1]);
            fr.mv_scale_x = float(mvs[0]), fr.mv_scale_y = float(mvs[1]);
            for (int i = 0; i < 16; ++i) fr.world_to_view[i] = float(w2v[i]), fr.view_to_clip[i] = float(v2c[i]);
            if (!rst.empty() && rst[0] != 0) fr.flags |= ONR_RR_FRAME_RESET;
            OnrRrTexture* t[8] = {&fr.color, &fr.diffuse_albedo, &fr.specular_albedo, &fr.normals, &fr.roughness, &fr.depth, &fr.motion,
                                  &fr.specular_hit_distance};
            for (int i = 0; i < 8; ++i) *t[i] = OnrRrTexture{tex[i].res, uint32_t(SRV), 0, 0};
            // ONR_VALIDATE_DM=1 / ONR_VALIDATE_SSS=1: the roughness texture (R16F) also given as the disocclusion mask /
            // the SSS guide (they run; NGX has no capture with them to compare)
            if (getenv("ONR_VALIDATE_DM")) fr.disocclusion_mask = OnrRrTexture{tex[4].res, uint32_t(SRV), 0, 0};
            if (getenv("ONR_VALIDATE_SSS")) fr.sss_guide = OnrRrTexture{tex[4].res, uint32_t(SRV), 0, 0};
            // ONR_VALIDATE_DM_INVERT=1: the mask read as 1 - mask (the host's diagnostic switch)
            if (getenv("ONR_VALIDATE_DM_INVERT")) fr.flags |= ONR_RR_FRAME_DM_INVERT;
            if (getenv("ONR_VALIDATE_DM_CLAMP")) fr.flags |= ONR_RR_FRAME_DM_CLAMP;
            // ONR_VALIDATE_DM_SCALE=s: the host's mask scale (with DM_SHIFT=12, s = 1/4096 must give the plain mask)
            if (getenv("ONR_VALIDATE_DM_SCALE")) fr.disocclusion_mask_scale = float(atof(getenv("ONR_VALIDATE_DM_SCALE")));
            if (dm_shift && fr.disocclusion_mask.resource) fr.disocclusion_mask.resource = dmt.res;
            if (re9) fr.roughness = {};   // in normals.w
            fr.output = OnrRrTexture{out.res, uint32_t(out_state), 0, 0};
            // ONR_VALIDATE_TL=1: the colour also given as the transparency layer (its alpha lays it over everything);
            // ONR_VALIDATE_V1=1: a host that knows only the frame without the transparency fields
            if (getenv("ONR_VALIDATE_TL")) fr.transparency_layer = *t[0];
            if (tl_sparse >= 0) fr.transparency_layer = OnrRrTexture{tlt.res, uint32_t(SRV), 0, 0};
            // the V4 inputs (docs/NOTA-ENTRADAS.md). ONR_VALIDATE_NO_HIT=1: no specular hit distance (NGX's run without it is
            // the capture A_pan_..._no_specular_hit_distance). ONR_VALIDATE_RAY=1: the hit distance given as the ray direction
            // + hit distance instead (rr2 reads none of it: the NO_HIT hashes). ONR_VALIDATE_SMV=1: the motion vectors also as
            // GBuffer.SpecularMvec (the reflections then move as the pixel: the NO_HIT hashes). ONR_VALIDATE_CBT=1: the colour as
            // ColorBeforeTransparency (attribute 30 is 0: the hashes without it); =2: the normals. ONR_VALIDATE_RESP /
            // ONR_VALIDATE_DOF=1: the roughness as the responsivity mask / the depth of field guide; =0 a zero texture.
            // ONR_VALIDATE_TLO=2: the specular albedo as the layer's opacity.
            auto envv = [](const char* k) { const char* v = getenv(k); return v ? atoi(v) : -1; };
            if (envv("ONR_VALIDATE_NO_HIT") == 1) fr.specular_hit_distance = {};
            if (envv("ONR_VALIDATE_RAY") == 1) fr.specular_ray_direction_hit_distance = fr.specular_hit_distance, fr.specular_hit_distance = {};
            if (envv("ONR_VALIDATE_SMV") == 1) fr.specular_motion = fr.motion;
            if (const int v = envv("ONR_VALIDATE_CBT"); v > 0) fr.color_before_transparency = v == 1 ? fr.color : fr.normals;
            if (const int v = envv("ONR_VALIDATE_RESP"); v >= 0) fr.responsivity_mask = OnrRrTexture{v ? tex[4].res : zt.res, uint32_t(SRV), 0, 0};
            if (const int v = envv("ONR_VALIDATE_DOF"); v >= 0) fr.depth_of_field_guide = OnrRrTexture{v ? tex[4].res : zt.res, uint32_t(SRV), 0, 0};
            if (tot.res) fr.transparency_opacity = OnrRrTexture{tot.res, uint32_t(SRV), 0, 0};
            if (envv("ONR_VALIDATE_TLO") == 2) fr.transparency_opacity = fr.specular_albedo;
            if (tmv.res) fr.transparency_motion = OnrRrTexture{tmv.res, uint32_t(SRV), 0, 0};
            if (getenv("ONR_VALIDATE_V1")) fr.size = ONR_RR_FRAME_SIZE_V1;
            if (getenv("ONR_VALIDATE_V4")) fr.size = ONR_RR_FRAME_SIZE_V4;
            void* job = nullptr;
            if (load) {
                D3D12_RESOURCE_BARRIER lb{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION};
                lb.Transition = {lsrc, 0, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE};
                cl[0]->ResourceBarrier(1, &lb);
                for (int i = 0; i < load; ++i) cl[0]->CopyBufferRegion(ldst, 0, lsrc, 0, 256ull << 20);
                std::swap(lb.Transition.StateBefore, lb.Transition.StateAfter);
                cl[0]->ResourceBarrier(1, &lb);
            }
            cl[0]->EndQuery(ph, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            uint32_t now = ONR_RR_BACKEND_VULKAN;
            if (a4.GetBackend) a4.GetBackend(ctx, &now);
            if (now == ONR_RR_BACKEND_D3D12) {
                // the whole frame on the game's list; a hop stops at the image, the host's list, the rest on the second list
                OnrRrDenoisedImage im{sizeof im};
                if (half) im.format = DXGI_FORMAT_R16G16B16A16_FLOAT;
                if (a4.Record(ctx, &fr, cl[0], edit > 0 ? uint32_t(edit) : 0u, edit > 0 ? &im : nullptr, &job)) {
                    api.GetLastError(err, sizeof err);
                    throw std::runtime_error(std::string("Record: ") + err);
                }
                cl[0]->EndQuery(ph, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                cl[0]->Close();
                al[1]->Reset();
                cl[1]->Reset(al[1], nullptr);
                cl[1]->EndQuery(ph, D3D12_QUERY_TYPE_TIMESTAMP, 2);
                if (job) {
                    hop_lists(im);
                    if (a4.RecordPost(ctx, job, cl[1], im.format)) {
                        api.GetLastError(err, sizeof err);
                        throw std::runtime_error(std::string("RecordPost: ") + err);
                    }
                }
                cl[1]->EndQuery(ph, D3D12_QUERY_TYPE_TIMESTAMP, 3);
                cl[1]->ResolveQueryData(ph, D3D12_QUERY_TYPE_TIMESTAMP, 0, 4, prb, 0);
                readback_output();
                ID3D12CommandList* first = cl[0];
                q->ExecuteCommandLists(1, &first);
                if (job && sharpen > 0 && !half) {
                    // the image read back, sharpened here, written back: an edit that follows the jitter as an NR does
                    wait_gpu();
                    auto* buf = static_cast<ID3D12Resource*>(im.buffer);
                    const UINT64 bytes = UINT64(im.row_pitch) * im.height;
                    if (!erb || erb->GetDesc().Width < bytes) {
                        if (erb) erb->Release(), eup->Release();
                        erb = make_buf(bytes, D3D12_HEAP_TYPE_READBACK), eup = make_buf(bytes, D3D12_HEAP_TYPE_UPLOAD);
                    }
                    auto copy = [&](ID3D12Resource* dst, ID3D12Resource* src) {
                        al[2]->Reset();
                        cl[2]->Reset(al[2], nullptr);
                        cl[2]->CopyBufferRegion(dst, 0, src, 0, bytes);
                        cl[2]->Close();
                        ID3D12CommandList* l = cl[2];
                        q->ExecuteCommandLists(1, &l);
                        wait_gpu();
                    };
                    copy(erb, buf);
                    uint8_t *src = nullptr, *dst = nullptr;
                    erb->Map(0, nullptr, reinterpret_cast<void**>(&src));
                    eup->Map(0, nullptr, reinterpret_cast<void**>(&dst));
                    const int W = int(im.width), H = int(im.height);
                    auto px = [&](int x, int y) { return reinterpret_cast<const float*>(src + UINT64(std::clamp(y, 0, H - 1)) * im.row_pitch) + 4 * std::clamp(x, 0, W - 1); };
                    for (int y = 0; y < H; ++y) {
                        float* o = reinterpret_cast<float*>(dst + UINT64(y) * im.row_pitch);
                        for (int x = 0; x < W; ++x)
                            for (int c = 0; c < 4; ++c) {
                                float s = 0;
                                for (int dy = -1; dy <= 1; ++dy)
                                    for (int dx = -1; dx <= 1; ++dx) s += px(x + dx, y + dy)[c];
                                const float v = px(x, y)[c];
                                o[4 * x + c] = c == 3 ? v : std::max(0.0f, v + sharpen * (v - s / 9.0f));
                            }
                    }
                    erb->Unmap(0, nullptr);
                    eup->Unmap(0, nullptr);
                    copy(buf, eup);
                } else if (job) {
                    ID3D12CommandList* h = cl[2];
                    q->ExecuteCommandLists(1, &h);
                }
                ID3D12CommandList* second = cl[1];
                q->ExecuteCommandLists(1, &second);
                wait_gpu();
                OnrRrInfo fi{sizeof fi};
                api.GetInfo(ctx, &fi);
                UINT64 g = 0, cpu = 0;
                q->GetClockCalibration(&g, &cpu);
                uint64_t* ts = nullptr;
                prb->Map(0, nullptr, reinterpret_cast<void**>(&ts));
                auto qpc = [&](uint64_t v) { return uint64_t(int64_t(cpu) + int64_t(double(int64_t(v - g)) * double(qpf.QuadPart) / double(tsf))); };
                if (fi.frames) path_times[fi.frames] = HostTimes{qpc(ts[0]), qpc(ts[1]), qpc(ts[2]), qpc(ts[3])};
                prb->Unmap(0, nullptr);
                d3d_frames = true;
                return;
            }
            if (api.RecordInputs(ctx, &fr, cl[0], &job)) {
                api.GetLastError(err, sizeof err);
                throw std::runtime_error(std::string("RecordInputs: ") + err);
            }
            cl[0]->EndQuery(ph, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            // ABI 2: the host's hop, recorded now: buffer -> texture -> buffer on the same queue
            OnrRrDenoisedImage im{sizeof im};
            if (half) im.format = DXGI_FORMAT_R16G16B16A16_FLOAT;
            if (edit > 0) {
                if (a2.PrepareDenoised(ctx, job, uint32_t(edit), &im)) throw std::runtime_error("PrepareDenoised");
                if (im.buffer) hop_lists(im);

            }
            cl[0]->Close();
            al[1]->Reset();
            cl[1]->Reset(al[1], nullptr);
            cl[1]->EndQuery(ph, D3D12_QUERY_TYPE_TIMESTAMP, 2);
            if (api.RecordOutput(ctx, job, cl[1])) throw std::runtime_error("RecordOutput");
            cl[1]->EndQuery(ph, D3D12_QUERY_TYPE_TIMESTAMP, 3);
            cl[1]->ResolveQueryData(ph, D3D12_QUERY_TYPE_TIMESTAMP, 0, 4, prb, 0);
            readback_output();

            ID3D12CommandList* first = cl[0];
            q->ExecuteCommandLists(1, &first);
            if (edit < 0) {
                LARGE_INTEGER e0, e1;
                QueryPerformanceCounter(&e0);
                if (api.Enqueue(ctx, job, q)) {
                    api.GetLastError(err, sizeof err);
                    throw std::runtime_error(std::string("Enqueue: ") + err);
                }
                QueryPerformanceCounter(&e1);
                enqueue_cpu += double(e1.QuadPart - e0.QuadPart) * 1000.0 / double(qpf.QuadPart), ++enqueue_n;
            } else {
                uint32_t exported = 0, format = 0;
                for (int i : {3, 4}) {
                    al[i]->Reset();
                    cl[i]->Reset(al[i], nullptr);
                    cl[i]->EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, i == 3 ? 0 : 3);
                    if (i == 4) cl[i]->ResolveQueryData(qh, D3D12_QUERY_TYPE_TIMESTAMP, 0, 4, qrb, 0);
                    cl[i]->Close();
                }
                ID3D12CommandList* pre = cl[3];
                q->ExecuteCommandLists(1, &pre);
                if (a2.EnqueueNetwork(ctx, job, q, &exported)) {
                    api.GetLastError(err, sizeof err);
                    throw std::runtime_error(std::string("EnqueueNetwork: ") + err);
                }
                if (exported && im.buffer) {
                    ID3D12CommandList* h = cl[2];
                    q->ExecuteCommandLists(1, &h);
                    format = im.format;
                }
                if (a2.EnqueuePost(ctx, job, q, format)) {
                    api.GetLastError(err, sizeof err);
                    throw std::runtime_error(std::string("EnqueuePost: ") + err);
                }
                ID3D12CommandList* post = cl[4];
                q->ExecuteCommandLists(1, &post);
                if (exported) ++hops;
            }
            ID3D12CommandList* second = cl[1];
            q->ExecuteCommandLists(1, &second);
            wait_gpu();
            {
                OnrRrInfo fi{sizeof fi};
                api.GetInfo(ctx, &fi);
                UINT64 g = 0, cpu = 0;
                q->GetClockCalibration(&g, &cpu);
                uint64_t* ts = nullptr;
                prb->Map(0, nullptr, reinterpret_cast<void**>(&ts));
                auto qpc = [&](uint64_t v) { return uint64_t(int64_t(cpu) + int64_t(double(int64_t(v - g)) * double(qpf.QuadPart) / double(tsf))); };
                if (fi.frames) path_times[fi.frames] = HostTimes{qpc(ts[0]), qpc(ts[1]), qpc(ts[2]), qpc(ts[3])};
                prb->Unmap(0, nullptr);
            }
            if (edit >= 0 && hops && a3.GetHopTiming) {
                UINT64 g = 0, cpu = 0;
                q->GetClockCalibration(&g, &cpu);
                uint64_t* ts = nullptr;
                qrb->Map(0, nullptr, reinterpret_cast<void**>(&ts));
                auto qpc = [&](uint64_t v) { return uint64_t(int64_t(cpu) + int64_t(double(int64_t(v - g)) * double(qpf.QuadPart) / double(tsf))); };
                host_times[hops] = HostTimes{qpc(ts[0]), qpc(ts[1]), qpc(ts[2]), qpc(ts[3])};
                qrb->Unmap(0, nullptr);
            }
        };

        // frame 0 until the network has run it (the frames before get the scaled colour while it builds)
        OnrRrInfo info{sizeof info};
        auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            run_frame(0);
            api.GetInfo(ctx, &info);
            if (info.frames > 0) break;
            if (info.state == ONR_RR_STATE_FAILED) throw std::runtime_error(std::string("runtime failed: ") + info.message);
            if (std::chrono::steady_clock::now() - t0 > std::chrono::minutes(5)) throw std::runtime_error("network not ready after 5 minutes");
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        printf("network ready after %.1f s (weights: %s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
               info.weights_source);
        printf("frame  PSNR dB  equal %%   GPU ms (ingest + network + pack)  output hash\n");
        int bad = 0;
        double lo = 1e9, hi = 0;
        for (int f = 0; f < nframes; ++f) {
            if (f) run_frame(f);
            api.GetInfo(ctx, &info);
            uint16_t* m = nullptr;
            rb->Map(0, nullptr, reinterpret_cast<void**>(&m));
            wchar_t p[MAX_PATH];
            swprintf_s(p, L"%s\\f%04d_output.raw", frames.c_str(), f);
            // a scene without an NGX capture (frames dir "-"): the output against itself
            std::vector<uint8_t> ref;
            if (wcscmp(frames.c_str(), L"-")) ref = read_file(p);
            else ref.assign(size_t(opitch) * OH, 0), memcpy(ref.data(), m, ref.size());
            const uint16_t* r = reinterpret_cast<const uint16_t*>(ref.data());
            double se = 0, peak = 0;
            size_t n = 0, eq = 0;
            uint64_t hash = 1469598103934665603ull;
            if (re9) {
                // the R9G9B9E5 words: the output's own, or the rgba16f output's encoding (reference)
                const uint32_t* m32 = reinterpret_cast<const uint32_t*>(m);
                for (size_t k = 0; k < size_t(OW) * OH; ++k) {
                    uint32_t e = m32[k];
                    if (re9 == 2) {
                        const float c[3] = {h2f(m[4 * k]), h2f(m[4 * k + 1]), h2f(m[4 * k + 2])};
                        e = rgb9e5_of(c);
                    }
                    hash = (hash ^ e) * 1099511628211ull;
                    for (int ch = 0; ch < 3; ++ch) {
                        double u = rgb9e5_ch(e, ch), v = h2f(r[4 * k + ch]);
                        se += (u - v) * (u - v), peak = std::max(peak, std::fabs(v)), ++n, eq += f2h_exact(u) == r[4 * k + ch];
                    }
                }
            } else
                for (size_t i = 0; i < size_t(OW) * OH * 4; ++i) {
                    hash = (hash ^ m[i]) * 1099511628211ull;
                    if (i % 4 == 3) continue;
                    double u = h2f(m[i]), v = h2f(r[i]);
                    se += (u - v) * (u - v), peak = std::max(peak, std::fabs(v)), ++n, eq += m[i] == r[i];
                }
            rb->Unmap(0, nullptr);
            double db = se > 0 ? 10 * std::log10(peak * peak / (se / double(n))) : 999;
            lo = std::min(lo, db), hi = std::max(hi, db);
            printf("%5d  %7.2f  %7.3f  %7.2f  %016llx\n", f, db, 100.0 * double(eq) / double(n), info.gpu_ms_last, (unsigned long long)hash);
            // ONR_VALIDATE_DUMP=<prefix>: every frame's output, rgba16f rows, as <prefix><frame>.raw
            if (const char* dp = getenv("ONR_VALIDATE_DUMP")) {
                char fn[512];
                snprintf(fn, sizeof fn, "%s%04d.raw", dp, f);
                if (FILE* df = fopen(fn, "wb")) rb->Map(0, nullptr, reinterpret_cast<void**>(&m)), fwrite(m, 1, size_t(opitch) * OH, df), rb->Unmap(0, nullptr), fclose(df);
            }
            bad += db < 45;
        }
        printf("PSNR %.2f .. %.2f dB over %d frames%s\n", lo, hi, nframes, bad ? " (some below 45 dB)" : "");
        // ONR_VALIDATE_LOOP=n: the last frame n more times without new guides, for the GPU time at a steady clock (the
        // runtime log splits it into ingest, network and pack every 600 frames)
        if (const char* lp = getenv("ONR_VALIDATE_LOOP")) {
            for (int i = 0, n = atoi(lp); i < n; ++i) run_frame(nframes - 1, false);
            api.GetInfo(ctx, &info);
            printf("loop of %s frames: GPU %.2f ms (average of the last 64: %.2f ms)\n", lp, info.gpu_ms_last, info.gpu_ms_average);
            // the hops the runtime still keeps (its last 16, less the ones whose submits it has not read back yet)
            double sum[7] = {};
            int n = 0;
            for (uint64_t h = hops > 16 ? hops - 16 : 1; h <= hops && a3.GetHopTiming; ++h) {
                OnrRrHopTiming ht{sizeof ht};
                auto it = host_times.find(h);
                if (it == host_times.end() || a3.GetHopTiming(ctx, h, &ht) != ONR_RR_OK) continue;
                const HostTimes& x = it->second;
                const double ms = 1000.0 / double(qpf.QuadPart);
                const double d[7] = {double(int64_t(ht.net_begin - x.pre)) * ms, double(int64_t(ht.net_end - ht.net_begin)) * ms,
                                     double(int64_t(x.h0 - ht.net_end)) * ms,    double(int64_t(x.h1 - x.h0)) * ms,
                                     double(int64_t(ht.post_begin - x.h1)) * ms, double(int64_t(ht.post_end - ht.post_begin)) * ms,
                                     double(int64_t(x.post - ht.post_end)) * ms};
                for (int i = 0; i < 7; ++i) sum[i] += d[i];
                ++n;
            }
            if (n) {
                printf("hop timeline, mean of %d frames (ms): queue -> network %.3f | network %.3f | -> host %.3f | host lists %.3f | "
                       "-> post %.3f | post %.3f | -> queue %.3f | total %.3f\n",
                       n, sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n, sum[4] / n, sum[5] / n, sum[6] / n,
                       (sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6]) / n);
            }
            // the path without a hop: the game's copies, the two queue crossings and the Vulkan span
            double ps[6] = {};
            int pn = 0;
            const double ms = 1000.0 / double(qpf.QuadPart);
            for (auto it = path_times.size() > 64 ? std::prev(path_times.end(), 64) : path_times.begin(); it != path_times.end(); ++it) {
                OnrRrHopTiming ht{sizeof ht};
                if (a3.GetHopTiming(ctx, it->first, &ht) != ONR_RR_OK) continue;
                const HostTimes& x = it->second;
                const double d[6] = {double(int64_t(x.h0 - x.pre)) * ms, double(int64_t(ht.net_begin - x.h0)) * ms,
                                     double(int64_t(ht.net_end - ht.net_begin)) * ms, double(int64_t(x.h1 - ht.net_end)) * ms,
                                     double(int64_t(x.post - x.h1)) * ms, double(int64_t(x.post - x.pre)) * ms};
                for (int i = 0; i < 6; ++i) ps[i] += d[i];
                ++pn;
            }
            if (pn)
                printf("path timeline, mean of %d frames (ms): inputs %.3f | queue -> vulkan %.3f | vulkan %.3f | -> queue %.3f | output %.3f | "
                       "total %.3f\n",
                       pn, ps[0] / pn, ps[1] / pn, ps[2] / pn, ps[3] / pn, ps[4] / pn, ps[5] / pn);
            // D3D12 backend: the whole path is what Record (and RecordPost) recorded into the game's lists
            if (d3d_frames) {
                double rec = 0, ps2 = 0, ws = 0;
                int dn = 0;
                for (auto it = path_times.size() > 64 ? std::prev(path_times.end(), 64) : path_times.begin(); it != path_times.end(); ++it) {
                    const HostTimes& x = it->second;
                    rec += double(int64_t(x.h0 - x.pre)) * ms, ps2 += double(int64_t(x.post - x.h1)) * ms, ws += double(int64_t(x.post - x.pre)) * ms;
                    ++dn;
                }
                if (dn)
                    printf("path timeline (D3D12, in the game's lists), mean of %d frames (ms): Record %.3f | RecordPost %.3f | first list start to "
                           "second list end %.3f\n",
                           dn, rec / dn, ps2 / dn, ws / dn);
            }
            if (enqueue_n) printf("Enqueue on the CPU: %.3f ms mean over %d calls\n", enqueue_cpu / enqueue_n, enqueue_n);
        }
        upload->Unmap(0, nullptr);
        if (idle) api.Destroy(idle);
        api.Destroy(ctx);
        return bad ? 1 : 0;
    } catch (const std::exception& e) {
        printf("ERROR: %s\n", e.what());
        return 1;
    }
}
