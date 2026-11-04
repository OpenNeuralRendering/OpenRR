// SPDX-License-Identifier: MIT
#include "d3d12_amd.hpp"

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace onr::d3d {

namespace {

// The AMD D3D12 driver extension, as exported by amdxc64.dll (AmdExtD3DCreateInterface). Only the vtable order and
// the two structures used here are declared; the layout follows the interface headers of the Radeon Developer Tool
// Suite samples (AmdDxExt).
struct __declspec(uuid("014937EC-9288-446F-A9AC-D75A8E3A984F")) AmdFactory : IUnknown {
    virtual HRESULT STDMETHODCALLTYPE CreateInterface(IUnknown* outer, REFIID riid, void** out) = 0;
};
struct AmdElfInfo {
    uint32_t type = 2;   // pipeline ELF
    void* next = nullptr;
    const void* elf = nullptr;
    size_t size = 0;
    uint32_t threads[3] = {1, 1, 1};
};
struct __declspec(uuid("BDC14598-B7D2-4A8D-9CA5-67848E2AF745")) AmdDevice5 : IUnknown {
    virtual HRESULT STDMETHODCALLTYPE CreateGraphicsPipelineState(const void*, const void*, REFIID, void**) = 0;
    virtual void STDMETHODCALLTYPE PushMarker(ID3D12GraphicsCommandList*, const char*) = 0;
    virtual void STDMETHODCALLTYPE PopMarker(ID3D12GraphicsCommandList*) = 0;
    virtual void STDMETHODCALLTYPE SetMarker(ID3D12GraphicsCommandList*, const char*) = 0;
    virtual HRESULT STDMETHODCALLTYPE CheckExtFeatureSupport(uint32_t, void*, size_t) const = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateComputePipelineState(const void*, const void*, REFIID, void**) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreatePipelineState(const void*, const void*, REFIID, void**) = 0;
    virtual void STDMETHODCALLTYPE SetPrimitiveTopology(ID3D12GraphicsCommandList*, uint32_t) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateComputePipelineFromElf(AmdElfInfo*, REFIID, void**) = 0;
    virtual void STDMETHODCALLTYPE SetKernelArguments(ID3D12GraphicsCommandList*, uint32_t, uint32_t, const void* const*) = 0;
};

void check(HRESULT hr, const char* what) {
    if (FAILED(hr)) {
        char b[128];
        snprintf(b, sizeof b, "%s failed (0x%08lx)", what, (unsigned long)hr);
        throw std::runtime_error(b);
    }
}

void barrier(ID3D12GraphicsCommandList* cl, ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
    D3D12_RESOURCE_BARRIER x{};
    x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    x.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, a, b};
    cl->ResourceBarrier(1, &x);
}

ID3D12Resource* buffer(ID3D12Device* dev, uint64_t size, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state, bool uav) {
    D3D12_HEAP_PROPERTIES hp{heap};
    D3D12_RESOURCE_DESC rd{D3D12_RESOURCE_DIMENSION_BUFFER, 0, std::max<uint64_t>(size, 256), 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                           D3D12_TEXTURE_LAYOUT_ROW_MAJOR, uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE};
    ID3D12Resource* r = nullptr;
    check(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&r)), "CreateCommittedResource");
    return r;
}

}  // namespace

// Vulkan (XGL) user data: entry 0 global table, entries 1..64 the push constant dwords 0..63, entries at or past the
// spill threshold read from the spill table at byte 4 * entry. D3D12: entries 0..3 belong to the driver and root
// parameter 0 starts at entry 4. The root constants are therefore the push constant dwords 3..n-1 followed by 0..2:
// every entry >= 4 keeps its Vulkan number (the spill offsets in the ISA stay valid) and only the user SGPRs that
// held entries 1..3 are pointed at the tail. Only the PAL metadata (msgpack) changes, in place.
std::vector<uint8_t> patch_elf(const std::vector<uint8_t>& vk, uint32_t pc) {
    std::vector<uint8_t> e = vk;
    auto find = [&](const char* key, size_t n) {
        auto k = reinterpret_cast<const uint8_t*>(key);
        auto it = std::search(e.begin(), e.end(), k, k + n);
        if (it == e.end() || std::search(it + 1, e.end(), k, k + n) != e.end()) throw std::runtime_error(std::string("ELF: one ") + (key + 1) + " expected");
        return size_t(it - e.begin()) + n;
    };
    size_t i = find("\xb2.user_data_reg_map", 19);
    if (e[i] != 0xdc) throw std::runtime_error("ELF: reg map is not an array16");
    uint32_t n = (e[i + 1] << 8) | e[i + 2];
    i += 3;
    for (uint32_t s = 0; s < n; ++s) {
        if (e[i] == 0xce) { i += 5; continue; }   // special entry (global table, spill table) or unused
        uint8_t v = e[i];
        if (v >= 0x80) throw std::runtime_error("ELF: reg map entry is not a fixint");
        if (v >= 1 && v <= 3) e[i] = uint8_t(4 + (pc - 3) + (v - 1));
        else if (v >= 4 + (pc - 3)) throw std::runtime_error("ELF: an SGPR reads past the push constants");
        ++i;
    }
    i = find("\xb0.user_data_limit", 17);
    if (e[i] >= 0x80) throw std::runtime_error("ELF: user_data_limit is not a fixint");
    e[i] = uint8_t(4 + pc);
    return e;
}

uint32_t elf_threads(const std::vector<uint8_t>& e) {
    const char key[] = "\xb7.threadgroup_dimensions";
    auto k = reinterpret_cast<const uint8_t*>(key);
    auto it = std::search(e.begin(), e.end(), k, k + 24);
    if (it == e.end() || it[24] != 0x93) throw std::runtime_error("ELF: no threadgroup_dimensions");
    uint32_t n = 1;
    for (int d = 0, i = 25; d < 3; ++d) {
        uint8_t t = it[i];
        uint32_t v = t < 0x80 ? t : t == 0xcc ? it[i + 1] : t == 0xcd ? (it[i + 1] << 8 | it[i + 2]) : 0;
        if (!v) throw std::runtime_error("ELF: unexpected threadgroup_dimensions encoding");
        i += t < 0x80 ? 1 : t == 0xcc ? 2 : 3;
        n *= v;
    }
    return n;
}

Device::~Device() {
    for (IUnknown* u : std::initializer_list<IUnknown*>{qrb_, qh_, fence_, cl_, ca_, q_, static_cast<IUnknown*>(ext_), dev_})
        if (u) u->Release();
}

bool Device::init(std::string& err) {
    try {
        IDXGIFactory6* fac = nullptr;
        check(CreateDXGIFactory2(0, IID_PPV_ARGS(&fac)), "CreateDXGIFactory2");
        IDXGIAdapter1* ad = nullptr;
        check(fac->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&ad)), "EnumAdapterByGpuPreference");
        fac->Release();
        DXGI_ADAPTER_DESC1 d;
        ad->GetDesc1(&d);
        if (d.VendorId != 0x1002) throw std::runtime_error("the high-performance adapter is not AMD");
        check(D3D12CreateDevice(ad, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev_)), "D3D12CreateDevice");
        ad->Release();
        auto create = reinterpret_cast<HRESULT(__cdecl*)(IUnknown*, REFIID, void**)>(
            GetProcAddress(GetModuleHandleA("amdxc64.dll"), "AmdExtD3DCreateInterface"));
        if (!create) throw std::runtime_error("amdxc64.dll has no AmdExtD3DCreateInterface");
        AmdFactory* f = nullptr;
        check(create(dev_, __uuidof(AmdFactory), (void**)&f), "AmdExtD3DCreateInterface");
        AmdDevice5* x = nullptr;
        HRESULT hr = f->CreateInterface(dev_, __uuidof(AmdDevice5), (void**)&x);
        f->Release();
        check(hr, "IAmdExtD3DDevice5");
        ext_ = x;
        D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
        check(dev_->CreateCommandQueue(&qd, IID_PPV_ARGS(&q_)), "CreateCommandQueue");
        check(dev_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&ca_)), "CreateCommandAllocator");
        check(dev_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, ca_, nullptr, IID_PPV_ARGS(&cl_)), "CreateCommandList");
        cl_->Close();
        check(dev_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)), "CreateFence");
        D3D12_QUERY_HEAP_DESC qhd{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 512};
        check(dev_->CreateQueryHeap(&qhd, IID_PPV_ARGS(&qh_)), "CreateQueryHeap");
        qrb_ = buffer(dev_, 512 * 8, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false);
        check(q_->GetTimestampFrequency(&ts_freq_), "GetTimestampFrequency");
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

void Device::execute() {
    check(cl_->Close(), "Close");
    ID3D12CommandList* l = cl_;
    q_->ExecuteCommandLists(1, &l);
    q_->Signal(fence_, ++fv_);
    fence_->SetEventOnCompletion(fv_, nullptr);
    check(dev_->GetDeviceRemovedReason(), "device removed");
}

void Device::copy(ID3D12Resource* dst, ID3D12Resource* src, uint64_t size, bool to_uav) {
    ca_->Reset();
    cl_->Reset(ca_, nullptr);
    ID3D12Resource* uav = to_uav ? dst : src;
    barrier(cl_, uav, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, to_uav ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COPY_SOURCE);
    cl_->CopyBufferRegion(dst, 0, src, 0, size);
    barrier(cl_, uav, to_uav ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    execute();
}

Buffer Device::create(uint64_t size) {
    Buffer b;
    b.res = buffer(dev_, size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
    b.va = b.res->GetGPUVirtualAddress();
    b.size = size;
    return b;
}

void Device::upload(const Buffer& b, const void* src, uint64_t size) {
    ID3D12Resource* up = buffer(dev_, size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false);
    void* p = nullptr;
    up->Map(0, nullptr, &p);
    memcpy(p, src, size);
    up->Unmap(0, nullptr);
    copy(b.res, up, size, true);
    up->Release();
}

void Device::download(const Buffer& b, void* dst, uint64_t size) {
    ID3D12Resource* rb = buffer(dev_, size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false);
    copy(rb, b.res, size, false);
    void* p = nullptr;
    rb->Map(0, nullptr, &p);
    memcpy(dst, p, size);
    rb->Unmap(0, nullptr);
    rb->Release();
}

void Device::destroy(Buffer& b) {
    if (b.res) b.res->Release();
    b = Buffer{};
}

Kernel Device::load_elf(const std::vector<uint8_t>& elf, uint32_t threads, uint32_t pc_dwords) {
    Kernel k;
    k.pc_dwords = pc_dwords;
    D3D12_ROOT_PARAMETER rp{};
    rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rp.Constants.Num32BitValues = pc_dwords;
    D3D12_ROOT_SIGNATURE_DESC rsd{1, &rp};
    ID3DBlob *blob = nullptr, *msg = nullptr;
    check(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &msg), "D3D12SerializeRootSignature");
    check(dev_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&k.rs)), "CreateRootSignature");
    blob->Release();
    AmdElfInfo info;
    info.elf = elf.data();
    info.size = elf.size();
    info.threads[0] = threads;
    check(static_cast<AmdDevice5*>(ext_)->CreateComputePipelineFromElf(&info, IID_PPV_ARGS(&k.pso)), "CreateComputePipelineFromElf");
    return k;
}

void Device::destroy(Kernel& k) {
    if (k.pso) k.pso->Release();
    if (k.rs) k.rs->Release();
    k = Kernel{};
}

std::vector<double> Device::run(const Kernel& k, const void* pc, uint32_t groups, int n) {
    std::vector<uint32_t> rc(k.pc_dwords);   // push constant dwords 3..n-1, then 0..2 (patch_elf)
    const uint32_t* w = static_cast<const uint32_t*>(pc);
    for (uint32_t i = 3; i < k.pc_dwords; ++i) rc[i - 3] = w[i];
    for (uint32_t i = 0; i < 3; ++i) rc[k.pc_dwords - 3 + i] = w[i];
    ca_->Reset();
    cl_->Reset(ca_, nullptr);
    cl_->SetComputeRootSignature(k.rs);
    cl_->SetPipelineState(k.pso);
    cl_->SetComputeRoot32BitConstants(0, k.pc_dwords, rc.data(), 0);
    D3D12_RESOURCE_BARRIER uav{D3D12_RESOURCE_BARRIER_TYPE_UAV};
    for (int i = 0; i < n; ++i) {
        if (i) cl_->ResourceBarrier(1, &uav);
        cl_->EndQuery(qh_, D3D12_QUERY_TYPE_TIMESTAMP, 2 * i);
        cl_->Dispatch(groups, 1, 1);
        cl_->EndQuery(qh_, D3D12_QUERY_TYPE_TIMESTAMP, 2 * i + 1);
    }
    cl_->ResolveQueryData(qh_, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2 * n, qrb_, 0);
    execute();
    uint64_t* t = nullptr;
    qrb_->Map(0, nullptr, (void**)&t);
    std::vector<double> ms(n);
    for (int i = 0; i < n; ++i) ms[i] = double(t[2 * i + 1] - t[2 * i]) * 1000.0 / double(ts_freq_);
    qrb_->Unmap(0, nullptr);
    return ms;
}

}  // namespace onr::d3d
