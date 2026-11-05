// SPDX-License-Identifier: MIT
// Drives OpenNrRrRuntime.dll through its Vulkan host path (ABI 5) as a Vulkan game would: a device of its own made
// through PrepareVkDevice / FinishVkDevice, the scene's guides as images, RecordVk on the "game's" command buffer, and
// each frame's output compared with NGX's for the same frame (hashes as opennr_rr_validate prints them).
//
// usage: opennr_rr_validate_vk <runtime.dll> <data dir> <nvngx_dlssd.dll> <scene dir> <capture frames dir> <WxH> [frames]
// ONR_VALIDATE_TL=1: the colour also as the transparency layer. ONR_VALIDATE_VK_GENERAL=1: the guides in GENERAL
// (copied where they are) instead of SHADER_READ_ONLY_OPTIMAL. ONR_VALIDATE_VK_DEPTH=1: depth in a D32_SFLOAT depth
// image (DEPTH_STENCIL_READ_ONLY_OPTIMAL), =2 in D32_SFLOAT_S8_UINT. ONR_VALIDATE_VK_CHAIN=split: the game's features
// in the extension structures, not in VkPhysicalDeviceVulkan1xFeatures. ONR_VALIDATE_LOOP=n: the last frame n more
// times without new guides (timing: the runtime's GPU time, and the host's timestamps around RecordVk).
// ONR_VALIDATE_VK_NOVIEWS=1: no views given (everything copied). ONR_VALIDATE_LOAD=n: n copies of 256 MB in the command buffer before the evaluate.
// ONR_VALIDATE_RENDER=WxH: the first feature renders at WxH (the guides' top left corner).
// ONR_VALIDATE_POISON=1: the guides' top quarter as a sky (depth 0, zero normals and albedos) plus inf / NaN texels
// in the colour and the motion.
// ONR_VALIDATE_VK_SECOND=1: a second feature as Indiana Jones makes one (426x240 -> 640x360, the scene's top left corner),
// created on another thread while the first one renders, its frames recorded into a command buffer of the game's
// compute queue and submitted beside the first one's, so the two networks run at the same time on two queues.
// ONR_VALIDATE_VK_CHURN=n (with SECOND): then n times both features released (the small one with its last frame
// recorded and not yet submitted) and made again at the other size (1129x635 / 376x211 and back), 120 network frames each.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "opennr_rr_api.h"

namespace {

void vk_check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(what) + " failed (" + std::to_string(int(r)) + ")");
}

std::vector<uint8_t> read_file(const std::wstring& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read a scene file");
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
}

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

struct Img {
    VkImage img = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkFormat fmt = VK_FORMAT_UNDEFINED;
    uint32_t w = 0, h = 0, bpp = 0;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;   // all aspects (barriers)
    VkImageAspectFlags copy_aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;          // where it rests between frames
    OnrRrVkImage desc{};
};

struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* map = nullptr;
};

}  // namespace

int wmain(int argc, wchar_t** argv) {
    if (argc < 7) {
        printf("usage: opennr_rr_validate_vk <runtime.dll> <data dir> <nvngx_dlssd.dll> <scene dir> <capture frames dir> <WxH> [frames]\n");
        return 2;
    }
    try {
        UINT OW = 0, OH = 0;
        swscanf_s(argv[6], L"%ux%u", &OW, &OH);
        const int nframes = argc > 7 ? _wtoi(argv[7]) : 12;
        std::wstring scene = argv[4], frames = argv[5];
        std::string js;
        {
            auto b = read_file(scene + L"\\sequence.json");
            js.assign(b.begin(), b.end());
        }
        auto rs = json_list(js, 0, "render_size");
        const uint32_t RW = uint32_t(rs[0]), RH = uint32_t(rs[1]);
        const size_t fr0 = js.find("\"frames\"");
        const bool general = getenv("ONR_VALIDATE_VK_GENERAL") != nullptr;
        const int depth_kind = getenv("ONR_VALIDATE_VK_DEPTH") ? atoi(getenv("ONR_VALIDATE_VK_DEPTH")) : 0;
        const bool split_chain = getenv("ONR_VALIDATE_VK_CHAIN") && !strcmp(getenv("ONR_VALIDATE_VK_CHAIN"), "split");
        const int load = getenv("ONR_VALIDATE_LOAD") ? std::max(0, atoi(getenv("ONR_VALIDATE_LOAD"))) : 0;
        const bool no_views = getenv("ONR_VALIDATE_VK_NOVIEWS") != nullptr;
        const bool poison = getenv("ONR_VALIDATE_POISON") != nullptr;

        HMODULE dll = LoadLibraryW(argv[1]);
        if (!dll) throw std::runtime_error("cannot load the runtime DLL");
        auto get = reinterpret_cast<PFN_OnrRrGetApi>(GetProcAddress(dll, "OnrRrGetApi"));
        OnrRrApi5 a5{};
        OnrRrApi& api = a5.v4.v3.v2.v1;
        if (!get || get(ONR_RR_ABI_VERSION_5, &api) || !a5.PrepareVkDevice || !a5.FinishVkDevice || !a5.RecordVk)
            throw std::runtime_error("the runtime has no ABI 5");
        char err[512] = {};

        // the "game": a Vulkan 1.3 instance and device of its own, as a game makes them
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "opennr-rr-validate-vk";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        VkInstance inst = VK_NULL_HANDLE;
        vk_check(vkCreateInstance(&ici, nullptr, &inst), "vkCreateInstance");
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(inst, &n, nullptr);
        std::vector<VkPhysicalDevice> pds(n);
        vkEnumeratePhysicalDevices(inst, &n, pds.data());
        VkPhysicalDevice pd = VK_NULL_HANDLE;
        for (auto p : pds) {
            VkPhysicalDeviceProperties pp;
            vkGetPhysicalDeviceProperties(p, &pp);
            if (pp.vendorID == 0x1002 && pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                pd = p;
                printf("Vulkan device: %s\n", pp.deviceName);
                break;
            }
        }
        if (!pd) throw std::runtime_error("no discrete AMD GPU");
        VkPhysicalDeviceIDProperties idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 pp2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &idp};
        vkGetPhysicalDeviceProperties2(pd, &pp2);
        const double ts_ns = pp2.properties.limits.timestampPeriod;
        if (api.Check(idp.deviceLUIDValid ? idp.deviceLUID : nullptr, argv[3])) {
            api.GetLastError(err, sizeof err);
            printf("Check: unsupported: %s\n", err);
            return 1;
        }
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qf(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf.data());
        uint32_t gfx = UINT32_MAX, comp = UINT32_MAX;
        for (uint32_t i = 0; i < nq; ++i) {
            if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && gfx == UINT32_MAX) gfx = i;
            else if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && comp == UINT32_MAX) comp = i;
        }
        // what a game asks for: its own features and queues (graphics + one async compute)
        float prio[2] = {1.0f, 1.0f};
        VkDeviceQueueCreateInfo qci[2] = {{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}, {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}};
        qci[0].queueFamilyIndex = gfx, qci[0].queueCount = 1, qci[0].pQueuePriorities = prio;
        qci[1].queueFamilyIndex = comp, qci[1].queueCount = 1, qci[1].pQueuePriorities = prio + 1;
        VkPhysicalDeviceVulkan13Features g13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        g13.dynamicRendering = VK_TRUE;
        VkPhysicalDeviceVulkan12Features g12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &g13};
        g12.timelineSemaphore = VK_TRUE;
        g12.descriptorIndexing = VK_TRUE;
        VkPhysicalDeviceTimelineSemaphoreFeatures gts{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
        gts.timelineSemaphore = VK_TRUE;
        VkPhysicalDeviceDynamicRenderingFeatures gdr{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES, &gts};
        gdr.dynamicRendering = VK_TRUE;
        VkPhysicalDeviceFeatures gcore{};
        gcore.samplerAnisotropy = VK_TRUE;
        VkDeviceCreateInfo game{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, split_chain ? static_cast<void*>(&gdr) : static_cast<void*>(&g12)};
        game.queueCreateInfoCount = comp == UINT32_MAX ? 1 : 2;
        game.pQueueCreateInfos = qci;
        game.pEnabledFeatures = &gcore;
        VkDeviceCreateInfo mine{};
        void* setup = nullptr;
        const int32_t prc = a5.PrepareVkDevice(pd, &game, &mine, &setup);
        if (prc) {
            api.GetLastError(err, sizeof err);
            throw std::runtime_error(std::string("PrepareVkDevice: ") + err);
        }
        VkDevice dev = VK_NULL_HANDLE;
        const VkResult cr = vkCreateDevice(pd, &mine, nullptr, &dev);
        uint32_t rfam = 0, ridx = 0;
        a5.FinishVkDevice(setup, cr == VK_SUCCESS ? dev : nullptr, &rfam, &ridx);
        vk_check(cr, "vkCreateDevice (with the runtime's setup)");
        if (g12.storageBuffer8BitAccess || gdr.pNext != &gts || gcore.shaderInt64) throw std::runtime_error("FinishVkDevice left the game's structures changed");
        printf("device: %u queue families asked, the runtime's queue: family %u index %u, %u extensions\n", mine.queueCreateInfoCount, rfam, ridx,
               mine.enabledExtensionCount);
        VkQueue q = VK_NULL_HANDLE;
        vkGetDeviceQueue(dev, gfx, 0, &q);
        VkPhysicalDeviceMemoryProperties memp;
        vkGetPhysicalDeviceMemoryProperties(pd, &memp);
        auto mem_type = [&](uint32_t bits, VkMemoryPropertyFlags want) {
            for (uint32_t i = 0; i < memp.memoryTypeCount; ++i)
                if ((bits & (1u << i)) && (memp.memoryTypes[i].propertyFlags & want) == want) return i;
            throw std::runtime_error("no memory type");
        };
        auto make_buf = [&](VkDeviceSize size, VkBufferUsageFlags usage, bool host) {
            Buf b;
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size = size;
            bci.usage = usage;
            vk_check(vkCreateBuffer(dev, &bci, nullptr, &b.buf), "vkCreateBuffer");
            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(dev, b.buf, &req);
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = req.size;
            mai.memoryTypeIndex = mem_type(req.memoryTypeBits, host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                                                    : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            vk_check(vkAllocateMemory(dev, &mai, nullptr, &b.mem), "vkAllocateMemory");
            vkBindBufferMemory(dev, b.buf, b.mem, 0);
            if (host) vk_check(vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map), "vkMapMemory");
            return b;
        };
        const uint32_t both[2] = {gfx, comp};
        bool shared_imgs = false;   // the second feature's images: both queues use them
        auto make_img = [&](uint32_t w, uint32_t h, VkFormat f, uint32_t bpp, VkImageUsageFlags usage, VkImageLayout rest) {
            Img im;
            im.fmt = f, im.w = w, im.h = h, im.bpp = bpp, im.layout = rest;
            if (f == VK_FORMAT_D32_SFLOAT) im.aspect = im.copy_aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (f == VK_FORMAT_D32_SFLOAT_S8_UINT) im.aspect = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, im.copy_aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
            VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ci.imageType = VK_IMAGE_TYPE_2D;
            ci.format = f;
            ci.extent = {w, h, 1};
            ci.mipLevels = ci.arrayLayers = 1;
            ci.samples = VK_SAMPLE_COUNT_1_BIT;
            ci.tiling = VK_IMAGE_TILING_OPTIMAL;
            ci.usage = usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            if (shared_imgs) ci.sharingMode = VK_SHARING_MODE_CONCURRENT, ci.queueFamilyIndexCount = 2, ci.pQueueFamilyIndices = both;
            vk_check(vkCreateImage(dev, &ci, nullptr, &im.img), "vkCreateImage");
            VkMemoryRequirements req;
            vkGetImageMemoryRequirements(dev, im.img, &req);
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = req.size;
            mai.memoryTypeIndex = mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            vk_check(vkAllocateMemory(dev, &mai, nullptr, &im.mem), "vkAllocateMemory");
            vkBindImageMemory(dev, im.img, im.mem, 0);
            // the view a game hands NGX: its format, the depth aspect of a depth format
            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.image = im.img;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = f;
            vci.subresourceRange = {im.copy_aspect, 0, 1, 0, 1};
            vk_check(vkCreateImageView(dev, &vci, nullptr, &im.view), "vkCreateImageView");
            im.desc = OnrRrVkImage{reinterpret_cast<uint64_t>(im.img), no_views ? 0 : reinterpret_cast<uint64_t>(im.view), uint32_t(f), w, h, 0, 0,
                                   uint32_t(ci.usage)};
            return im;
        };
        auto barrier = [&](VkCommandBuffer cb, const Img& im, VkImageLayout from, VkImageLayout to) {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask = b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
            b.oldLayout = from, b.newLayout = to;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = im.img;
            b.subresourceRange = {im.aspect, 0, 1, 0, 1};
            VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            di.imageMemoryBarrierCount = 1;
            di.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cb, &di);
        };

        const wchar_t* dirs[8] = {L"color", L"diffuse_albedo", L"specular_albedo", L"normals", L"roughness", L"depth", L"motion_vectors",
                                  L"specular_hit_distance"};
        VkFormat fmts[8] = {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT,
                            VK_FORMAT_R16_SFLOAT,          VK_FORMAT_R32_SFLOAT,     VK_FORMAT_R16G16_SFLOAT,  VK_FORMAT_R16_SFLOAT};
        const uint32_t bpps[8] = {8, 4, 4, 8, 2, 4, 4, 2};
        const VkImageLayout read = general ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        Img tex[8];
        for (int i = 0; i < 8; ++i) {
            VkFormat f = fmts[i];
            VkImageUsageFlags u = VK_IMAGE_USAGE_SAMPLED_BIT;
            VkImageLayout rest = read;
            if (i == 5 && depth_kind) {
                f = depth_kind == 2 ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_D32_SFLOAT;
                u |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
                rest = general ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            }
            tex[i] = make_img(RW, RH, f, bpps[i], u, rest);
        }
        Img out = make_img(OW, OH, VK_FORMAT_R16G16B16A16_SFLOAT, 8, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_GENERAL);
        const VkDeviceSize up_stride = VkDeviceSize(RW) * RH * 8;
        Buf up = make_buf(up_stride * 8, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
        Buf rb = make_buf(VkDeviceSize(OW) * OH * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
        Buf lsrc, ldst;
        if (load) {
            lsrc = make_buf(256ull << 20, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false);
            ldst = make_buf(256ull << 20, VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
        }

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = gfx;
        VkCommandPool pool;
        vk_check(vkCreateCommandPool(dev, &pci, nullptr, &pool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VkCommandBuffer cb;
        vk_check(vkAllocateCommandBuffers(dev, &cai, &cb), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence;
        vk_check(vkCreateFence(dev, &fci, nullptr, &fence), "vkCreateFence");
        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = 2;
        VkQueryPool qp;
        vk_check(vkCreateQueryPool(dev, &qpi, nullptr, &qp), "vkCreateQueryPool");
        auto submit_wait = [&] {
            vk_check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            vk_check(vkQueueSubmit(q, 1, &si, fence), "vkQueueSubmit");
            vk_check(vkWaitForFences(dev, 1, &fence, VK_TRUE, 30ull * 1000 * 1000 * 1000), "vkWaitForFences");
            vkResetFences(dev, 1, &fence);
        };
        auto begin = [&] {
            vkResetCommandBuffer(cb, 0);
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cb, &bi);
        };
        // the images into their resting layouts
        begin();
        for (auto& t : tex) barrier(cb, t, VK_IMAGE_LAYOUT_UNDEFINED, t.layout);
        barrier(cb, out, VK_IMAGE_LAYOUT_UNDEFINED, out.layout);
        submit_wait();

        OnrRrVkDevice vd{sizeof vd, VK_API_VERSION_1_3, inst, pd, dev, rfam, ridx};
        OnrRrCreateInfo ci{sizeof ci, ONR_RR_MODE_FAST, nullptr, RW, RH, OW, OH, argv[3], nullptr, argv[2], nullptr, ONR_RR_BACKEND_CHOICE_AUTO, &vd};
        if (const char* e = getenv("ONR_VALIDATE_RENDER")) {   // WxH: a smaller render size, the guides' top left corner
            UINT w = 0, h = 0;
            if (sscanf_s(e, "%ux%u", &w, &h) == 2 && w && h && w <= RW && h <= RH) ci.max_render_width = w, ci.max_render_height = h;
        }
        void* ctx = nullptr;
        if (api.Create(&ci, &ctx)) {
            api.GetLastError(err, sizeof err);
            throw std::runtime_error(std::string("Create: ") + err);
        }
        uint32_t backend = 0;
        a5.v4.GetBackend(ctx, &backend);
        // ONR_VALIDATE_FAST=1: the runtime's fast mode from the start; ONR_VALIDATE_FAST_FLIP=n: flipped every n frames
        auto set_fast = reinterpret_cast<int32_t (*)(void*, int32_t)>(GetProcAddress(dll, "OnrRrSetFastMode"));
        const int fast_flip = getenv("ONR_VALIDATE_FAST_FLIP") ? atoi(getenv("ONR_VALIDATE_FAST_FLIP")) : 0;
        if (getenv("ONR_VALIDATE_FAST") && set_fast) set_fast(ctx, 1);
        printf("render %ux%u -> %ux%u, %d frames, _f, backend %s, guides %s, depth %s\n", RW, RH, OW, OH, nframes,
               backend == ONR_RR_BACKEND_VULKAN_HOST ? "Vulkan host" : "other", general ? "GENERAL" : "SHADER_READ_ONLY_OPTIMAL",
               depth_kind == 2 ? "D32_SFLOAT_S8_UINT" : depth_kind ? "D32_SFLOAT" : "R32_SFLOAT");
        if (backend != ONR_RR_BACKEND_VULKAN_HOST) throw std::runtime_error("not the Vulkan host backend");

        // the second feature (ONR_VALIDATE_VK_SECOND)
        const bool second = getenv("ONR_VALIDATE_VK_SECOND") && comp != UINT32_MAX;
        const uint32_t RW2 = 426, RH2 = 240, OW2 = 640, OH2 = 360;
        uint32_t cw = ci.max_render_width, chh = ci.max_render_height, sw2 = RW2, sh2 = RH2;   // the render sizes recorded (CHURN changes them)
        Img tex2[8], out2;
        VkQueue qc = VK_NULL_HANDLE;
        VkCommandPool pool2 = VK_NULL_HANDLE;
        VkCommandBuffer cb2 = VK_NULL_HANDLE;
        VkFence fence2 = VK_NULL_HANDLE;
        void* ctx2 = nullptr;
        std::thread maker;
        struct Joined {
            std::thread& t;
            ~Joined() { if (t.joinable()) t.join(); }
        } joined{maker};   // an error while it runs is reported, not std::terminate
        std::atomic<int> made{0};   // 1 created, -1 failed
        std::string made_err;
        int frames2 = 0;
        if (second) {
            shared_imgs = true;
            for (int i = 0; i < 8; ++i) tex2[i] = make_img(RW2, RH2, fmts[i], bpps[i], VK_IMAGE_USAGE_SAMPLED_BIT, read);
            out2 = make_img(OW2, OH2, VK_FORMAT_R16G16B16A16_SFLOAT, 8, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_GENERAL);
            shared_imgs = false;
            vkGetDeviceQueue(dev, comp, 0, &qc);
            VkCommandPoolCreateInfo p2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            p2.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            p2.queueFamilyIndex = comp;
            vk_check(vkCreateCommandPool(dev, &p2, nullptr, &pool2), "vkCreateCommandPool");
            VkCommandBufferAllocateInfo a2{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            a2.commandPool = pool2;
            a2.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            a2.commandBufferCount = 1;
            vk_check(vkAllocateCommandBuffers(dev, &a2, &cb2), "vkAllocateCommandBuffers");
            vk_check(vkCreateFence(dev, &fci, nullptr, &fence2), "vkCreateFence");
            // frame 0's guides, the top left corner, into the second feature's images (in their layouts)
            begin();
            for (int i = 0; i < 8; ++i) {
                wchar_t p[MAX_PATH];
                swprintf_s(p, L"%s\\%s\\%04d.raw", scene.c_str(), dirs[i], 0);
                auto d = read_file(p);
                memcpy(static_cast<uint8_t*>(up.map) + up_stride * i, d.data(), std::min<size_t>(d.size(), size_t(up_stride)));
                barrier(cb, tex2[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                VkBufferImageCopy r{};
                r.bufferOffset = up_stride * i;
                r.bufferRowLength = RW;
                r.imageSubresource = {tex2[i].copy_aspect, 0, 0, 1};
                r.imageExtent = {RW2, RH2, 1};
                vkCmdCopyBufferToImage(cb, up.buf, tex2[i].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
                barrier(cb, tex2[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, tex2[i].layout);
            }
            barrier(cb, out2, VK_IMAGE_LAYOUT_UNDEFINED, out2.layout);
            submit_wait();
        }
        auto record_second = [&](int f) {
            vkResetCommandBuffer(cb2, 0);
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cb2, &bi);
            OnrRrFrame fr{};
            fr.size = sizeof fr;
            fr.render_width = sw2, fr.render_height = sh2;
            fr.jitter_x = (f & 1) ? 0.25f : -0.25f, fr.jitter_y = (f & 2) ? 0.25f : -0.25f;
            fr.mv_scale_x = float(RW2), fr.mv_scale_y = float(RH2);
            for (int i = 0; i < 16; ++i) fr.world_to_view[i] = fr.view_to_clip[i] = (i % 5 == 0) ? 1.0f : 0.0f;
            OnrRrTexture* t[8] = {&fr.color, &fr.diffuse_albedo, &fr.specular_albedo, &fr.normals, &fr.roughness, &fr.depth, &fr.motion,
                                  &fr.specular_hit_distance};
            for (int i = 0; i < 8; ++i) *t[i] = OnrRrTexture{&tex2[i].desc, uint32_t(tex2[i].layout), 0, 0};
            fr.output = OnrRrTexture{&out2.desc, uint32_t(out2.layout), 0, 0};
            if (a5.RecordVk(ctx2, &fr, cb2)) {
                api.GetLastError(err, sizeof err);
                vkEndCommandBuffer(cb2);
                throw std::runtime_error(std::string("RecordVk (second feature): ") + err);
            }
            vk_check(vkEndCommandBuffer(cb2), "vkEndCommandBuffer");
        };

        double rec_ms = 0;
        int rec_n = 0;
        auto run_frame = [&](int f, bool upload_guides) {
            begin();
            if (load) {
                VkBufferCopy bc{0, 0, 256ull << 20};
                for (int i = 0; i < load; ++i) vkCmdCopyBuffer(cb, lsrc.buf, ldst.buf, 1, &bc);
            }
            for (int i = 0; i < 8 && upload_guides; ++i) {
                wchar_t p[MAX_PATH];
                swprintf_s(p, L"%s\\%s\\%04d.raw", scene.c_str(), dirs[i], f);
                auto d = read_file(p);
                memcpy(static_cast<uint8_t*>(up.map) + up_stride * i, d.data(), std::min<size_t>(d.size(), size_t(up_stride)));
                if (poison) {
                    // the top quarter as a sky a game may hand over: depth 0 (reversed-Z far), zero normals and
                    // albedos, and a sprinkle of inf / NaN in the colour and the motion
                    uint8_t* b = static_cast<uint8_t*>(up.map) + up_stride * i;
                    const size_t n = size_t(RW) * (RH / 4);
                    if (i == 1 || i == 2 || i == 3 || i == 5) memset(b, 0, n * bpps[i]);
                    if (i == 0 || i == 6) {
                        uint16_t* h = reinterpret_cast<uint16_t*>(b);
                        const size_t ch = bpps[i] / 2;
                        for (size_t k = 0; k < size_t(RW) * RH; k += 997) h[k * ch] = (k / 997) & 1 ? 0x7C00 : 0x7E00;
                    }
                }
                barrier(cb, tex[i], tex[i].layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                VkBufferImageCopy r{};
                r.bufferOffset = up_stride * i;
                r.imageSubresource = {tex[i].copy_aspect, 0, 0, 1};
                r.imageExtent = {RW, RH, 1};
                vkCmdCopyBufferToImage(cb, up.buf, tex[i].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
                barrier(cb, tex[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, tex[i].layout);
            }
            size_t at = fr0;
            for (int k = 0; k <= f; ++k) at = js.find("\"jitter\"", at + 1);
            OnrRrFrame fr{};
            fr.size = sizeof fr;
            auto jit = json_list(js, at, "jitter"), mvs = json_list(js, at, "mv_scale"), rst = json_list(js, at, "reset");
            auto w2v = json_list(js, at, "world_to_view"), v2c = json_list(js, at, "view_to_clip");
            fr.render_width = cw, fr.render_height = chh;
            fr.jitter_x = float(jit[0]), fr.jitter_y = float(jit[1]);
            fr.mv_scale_x = float(mvs[0]), fr.mv_scale_y = float(mvs[1]);
            for (int i = 0; i < 16; ++i) fr.world_to_view[i] = float(w2v[i]), fr.view_to_clip[i] = float(v2c[i]);
            if (!rst.empty() && rst[0] != 0) fr.flags |= ONR_RR_FRAME_RESET;
            OnrRrTexture* t[8] = {&fr.color, &fr.diffuse_albedo, &fr.specular_albedo, &fr.normals, &fr.roughness, &fr.depth, &fr.motion,
                                  &fr.specular_hit_distance};
            for (int i = 0; i < 8; ++i) *t[i] = OnrRrTexture{&tex[i].desc, uint32_t(tex[i].layout), 0, 0};
            fr.output = OnrRrTexture{&out.desc, uint32_t(out.layout), 0, 0};
            if (getenv("ONR_VALIDATE_TL")) fr.transparency_layer = fr.color;
            // the extra guides as opennr_rr_validate gives them (SSS, reflection MVs, colour before transparency, DoF)
            if (getenv("ONR_VALIDATE_SSS")) fr.sss_guide = *t[4];
            if (getenv("ONR_VALIDATE_SMV")) fr.specular_motion = fr.motion;
            if (getenv("ONR_VALIDATE_CBT")) fr.color_before_transparency = fr.color;
            if (getenv("ONR_VALIDATE_DOF")) fr.depth_of_field_guide = *t[4];
            vkCmdResetQueryPool(cb, qp, 0, 2);
            vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, qp, 0);
            if (a5.RecordVk(ctx, &fr, cb)) {
                api.GetLastError(err, sizeof err);
                vkEndCommandBuffer(cb);
                throw std::runtime_error(std::string("RecordVk: ") + err);
            }
            vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, qp, 1);
            // the output into the readback buffer
            barrier(cb, out, out.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy r{};
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            r.imageExtent = {OW, OH, 1};
            vkCmdCopyImageToBuffer(cb, out.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buf, 1, &r);
            barrier(cb, out, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out.layout);
            // the second feature's frame on the compute queue, running beside this one
            const bool two = second && made.load() == 1;
            if (two) {
                record_second(f);
                VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                si.commandBufferCount = 1;
                si.pCommandBuffers = &cb2;
                vk_check(vkQueueSubmit(qc, 1, &si, fence2), "vkQueueSubmit (compute)");
            }
            submit_wait();
            if (two) {
                vk_check(vkWaitForFences(dev, 1, &fence2, VK_TRUE, 30ull * 1000 * 1000 * 1000), "vkWaitForFences (second feature)");
                vkResetFences(dev, 1, &fence2);
                ++frames2;
            }
            uint64_t ts[2] = {};
            if (vkGetQueryPoolResults(dev, qp, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
                rec_ms += double(ts[1] - ts[0]) * ts_ns * 1e-6, ++rec_n;
        };

        OnrRrInfo info{sizeof info};
        auto t0 = std::chrono::steady_clock::now();
        if (second) {
            // created on another thread while the first feature renders, as the game does
            maker = std::thread([&] {
                OnrRrCreateInfo c2 = ci;
                c2.max_render_width = RW2, c2.max_render_height = RH2, c2.output_width = OW2, c2.output_height = OH2;
                if (api.Create(&c2, &ctx2)) {
                    char e2[512] = {};
                    api.GetLastError(e2, sizeof e2);
                    made_err = e2;
                    made = -1;
                } else {
                    made = 1;
                }
            });
        }
        for (;;) {
            run_frame(0, true);
            api.GetInfo(ctx, &info);
            if (info.frames > 0) break;
            if (info.state == ONR_RR_STATE_FAILED) throw std::runtime_error(std::string("runtime failed: ") + info.message);
            if (std::chrono::steady_clock::now() - t0 > std::chrono::minutes(5)) throw std::runtime_error("network not ready after 5 minutes");
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        printf("network ready after %.1f s (weights: %s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
               info.weights_source);
        if (second) {
            maker.join();
            if (made.load() < 0) throw std::runtime_error("Create (second feature): " + made_err);
            printf("second feature created (426x240 -> 640x360, compute queue)\n");
        }
        printf("frame  PSNR dB  equal %%   GPU ms (ingest + network + pack)  output hash\n");
        int bad = 0;
        double lo = 1e9, hi = 0;
        for (int f = 0; f < nframes; ++f) {
            if (fast_flip && set_fast && f % fast_flip == 0) set_fast(ctx, (f / fast_flip) % 2 == 0 ? 1 : 0);
            if (f) run_frame(f, true);
            api.GetInfo(ctx, &info);
            const uint16_t* m = static_cast<const uint16_t*>(rb.map);
            wchar_t p[MAX_PATH];
            swprintf_s(p, L"%s\\f%04d_output.raw", frames.c_str(), f);
            auto ref = read_file(p);
            const uint16_t* r = reinterpret_cast<const uint16_t*>(ref.data());
            double se = 0, peak = 0;
            size_t cnt = 0, eq = 0;
            uint64_t hash = 1469598103934665603ull;
            for (size_t i = 0; i < size_t(OW) * OH * 4; ++i) {
                hash = (hash ^ m[i]) * 1099511628211ull;
                if (i % 4 == 3) continue;
                double u = h2f(m[i]), v = h2f(r[i]);
                se += (u - v) * (u - v), peak = std::max(peak, std::fabs(v)), ++cnt, eq += m[i] == r[i];
            }
            double db = se > 0 ? 10 * std::log10(peak * peak / (se / double(cnt))) : 999;
            lo = std::min(lo, db), hi = std::max(hi, db);
            printf("%5d  %7.2f  %7.3f  %7.2f  %016llx\n", f, db, 100.0 * double(eq) / double(cnt), info.gpu_ms_last, (unsigned long long)hash);
            bad += db < 45;
        }
        printf("PSNR %.2f .. %.2f dB over %d frames%s\n", lo, hi, nframes, bad ? " (some below 45 dB)" : "");
        if (const char* lp = getenv("ONR_VALIDATE_LOOP")) {
            rec_ms = 0, rec_n = 0;
            for (int i = 0, k = atoi(lp); i < k; ++i) run_frame(nframes - 1, false);
            api.GetInfo(ctx, &info);
            printf("loop of %s frames: GPU %.2f ms (average of the last 64: %.2f ms); RecordVk on the command buffer, host timestamps: %.3f ms mean\n",
                   lp, info.gpu_ms_last, info.gpu_ms_average, rec_n ? rec_ms / rec_n : 0.0);
        }
        if (const char* churn = getenv("ONR_VALIDATE_VK_CHURN"); churn && second && made.load() == 1) {
            for (int k = 0, n = atoi(churn); k < n; ++k) {
                // the small feature released with its last frame recorded, submitted after the release
                record_second(0);
                api.Destroy(ctx2);
                ctx2 = nullptr;
                made = 0;
                VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                si.commandBufferCount = 1;
                si.pCommandBuffers = &cb2;
                vk_check(vkQueueSubmit(qc, 1, &si, fence2), "vkQueueSubmit (compute)");
                vk_check(vkWaitForFences(dev, 1, &fence2, VK_TRUE, 30ull * 1000 * 1000 * 1000), "vkWaitForFences (released feature)");
                vkResetFences(dev, 1, &fence2);
                vkDeviceWaitIdle(dev);
                api.Destroy(ctx);
                ctx = nullptr;
                const bool alt = (k & 1) == 0;
                cw = alt ? 1129 : RW, chh = alt ? 635 : RH, sw2 = alt ? 376 : RW2, sh2 = alt ? 211 : RH2;
                OnrRrCreateInfo c1 = ci;
                c1.max_render_width = cw, c1.max_render_height = chh;
                if (api.Create(&c1, &ctx)) {
                    api.GetLastError(err, sizeof err);
                    throw std::runtime_error(std::string("Create (churn): ") + err);
                }
                maker = std::thread([&] {
                    OnrRrCreateInfo c2 = ci;
                    c2.max_render_width = sw2, c2.max_render_height = sh2, c2.output_width = sw2 * 3 / 2, c2.output_height = sh2 * 3 / 2;
                    if (api.Create(&c2, &ctx2)) {
                        char e2[512] = {};
                        api.GetLastError(e2, sizeof e2);
                        made_err = e2;
                        made = -1;
                    } else {
                        made = 1;
                    }
                });
                const int f0 = frames2;
                const auto c0 = std::chrono::steady_clock::now();
                OnrRrInfo a{sizeof a}, b{sizeof b};
                for (int i = 0;; ++i) {
                    run_frame(i % nframes, false);
                    if (made.load() == 1) {
                        api.GetInfo(ctx, &a);
                        api.GetInfo(ctx2, &b);
                        // a small view runs no network (ONR_RR_SMALL_VIEWS=1 runs it): 120 frames of its scaled colour
                        const bool small = !strncmp(b.message, "small view", 10);
                        if (a.frames >= 120 && (small ? frames2 - f0 >= 120 : b.frames >= 120)) break;
                    }
                    if (std::chrono::steady_clock::now() - c0 > std::chrono::minutes(2)) throw std::runtime_error("churn: networks not running after 2 minutes");
                }
                maker.join();
                if (made.load() < 0) throw std::runtime_error("Create (churn, second feature): " + made_err);
                printf("churn %d: %ux%u and %ux%u, %d frames of both, network frames %u / %u, GPU %.2f / %.2f ms\n", k + 1, cw, chh, sw2, sh2, frames2 - f0,
                       a.frames, b.frames, a.gpu_ms_last, b.gpu_ms_last);
            }
        }
        if (second) {
            OnrRrInfo i2{sizeof i2};
            api.GetInfo(ctx2, &i2);
            printf("second feature: %d frames recorded beside the first, %u network frames, state %u, GPU %.2f ms\n", frames2, i2.frames, i2.state,
                   i2.gpu_ms_last);
        }
        // the device idle before Destroy, as the host promises
        vkDeviceWaitIdle(dev);
        api.Destroy(ctx);
        if (ctx2) api.Destroy(ctx2);
        if (second) {
            vkDestroyFence(dev, fence2, nullptr);
            vkDestroyCommandPool(dev, pool2, nullptr);
            for (auto& t : tex2) vkDestroyImageView(dev, t.view, nullptr), vkDestroyImage(dev, t.img, nullptr), vkFreeMemory(dev, t.mem, nullptr);
            vkDestroyImageView(dev, out2.view, nullptr), vkDestroyImage(dev, out2.img, nullptr), vkFreeMemory(dev, out2.mem, nullptr);
        }
        vkDestroyQueryPool(dev, qp, nullptr);
        vkDestroyFence(dev, fence, nullptr);
        vkDestroyCommandPool(dev, pool, nullptr);
        for (auto* b : {&up, &rb, &lsrc, &ldst})
            if (b->buf) vkDestroyBuffer(dev, b->buf, nullptr), vkFreeMemory(dev, b->mem, nullptr);
        for (auto& t : tex) vkDestroyImageView(dev, t.view, nullptr), vkDestroyImage(dev, t.img, nullptr), vkFreeMemory(dev, t.mem, nullptr);
        vkDestroyImageView(dev, out.view, nullptr), vkDestroyImage(dev, out.img, nullptr), vkFreeMemory(dev, out.mem, nullptr);
        vkDestroyDevice(dev, nullptr);
        vkDestroyInstance(inst, nullptr);
        return bad ? 1 : 0;
    } catch (const std::exception& e) {
        printf("ERROR: %s\n", e.what());
        return 1;
    }
}
