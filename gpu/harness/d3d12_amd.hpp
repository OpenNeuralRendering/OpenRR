// SPDX-License-Identifier: MIT
// Prototype (FASE9-D3D12): our Vulkan kernels on a plain D3D12 device. The pipeline ELF that the AMD Vulkan driver
// compiled (LLPC, WMMA and all) is handed to the AMD D3D12 driver through its extension interface
// (IAmdExtD3DDevice5::CreateComputePipelineFromElf); push constants travel as root constants. Buffers are reached
// by GPU virtual address, as in Vulkan.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct ID3D12Device;
struct ID3D12CommandQueue;
struct ID3D12PipelineState;
struct ID3D12RootSignature;
struct ID3D12Resource;
struct ID3D12CommandAllocator;
struct ID3D12GraphicsCommandList;
struct ID3D12Fence;
struct ID3D12QueryHeap;

namespace onr::d3d {

struct Kernel {
    ID3D12PipelineState* pso = nullptr;
    ID3D12RootSignature* rs = nullptr;
    uint32_t pc_dwords = 0;
};

struct Buffer {
    ID3D12Resource* res = nullptr;
    uint64_t va = 0, size = 0;
};

// Vulkan pipeline ELF -> D3D12 user data layout (see the .cpp). Throws if the ELF does not fit the scheme.
std::vector<uint8_t> patch_elf(const std::vector<uint8_t>& vk_elf, uint32_t pc_dwords);
uint32_t elf_threads(const std::vector<uint8_t>& elf);   // threads per workgroup (our kernels are 1D)

class Device {
public:
    ~Device();
    bool init(std::string& err);
    ID3D12Device* d3d() const { return dev_; }
    Buffer create(uint64_t size);   // default heap, UAV, left in UNORDERED_ACCESS
    void upload(const Buffer& b, const void* src, uint64_t size);
    void download(const Buffer& b, void* dst, uint64_t size);
    void destroy(Buffer& b);
    Kernel load_elf(const std::vector<uint8_t>& elf, uint32_t threads, uint32_t pc_dwords);
    void destroy(Kernel& k);
    // one command list: n dispatches with a UAV barrier between them, a timestamp around each; returns ms per dispatch
    std::vector<double> run(const Kernel& k, const void* pc, uint32_t groups, int n);

private:
    void copy(ID3D12Resource* dst, ID3D12Resource* src, uint64_t size, bool to_uav);
    void execute();

    ID3D12Device* dev_ = nullptr;
    ID3D12CommandQueue* q_ = nullptr;
    ID3D12CommandAllocator* ca_ = nullptr;
    ID3D12GraphicsCommandList* cl_ = nullptr;
    ID3D12Fence* fence_ = nullptr;
    ID3D12QueryHeap* qh_ = nullptr;
    ID3D12Resource* qrb_ = nullptr;
    void* ext_ = nullptr;   // IAmdExtD3DDevice5
    uint64_t fv_ = 0, ts_freq_ = 1;
};

}  // namespace onr::d3d
