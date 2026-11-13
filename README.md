# OpenRR

OpenRR runs NVIDIA's DLSS Ray Reconstruction network (`nvngx_dlssd.dll` 310.9.1, preset F) on AMD Radeon RX 9000
cards (RDNA4), with kernels written from scratch. It is the Ray Reconstruction runtime of the OpenNR project and runs
inside the [OpenNR OptiScaler fork](https://github.com/MatheusFerreiraS/OpenNR-OptiScaler): when a game asks for DLSS
Ray Reconstruction, OptiScaler hands the frame to OpenRR.

Version 0.1.6. MIT licence.

Discord: <https://discord.gg/wYhvS3JSHM>

Ko-fi: <https://ko-fi.com/T6T213OVFE>

## Requirements

- An RX 9000 card (RDNA4). RX 7000 cards and RDNA3 / RDNA 3.5 graphics (Radeon 780M, 890M, 8060S) are experimental, with kernels
  built for RDNA3. Other cards are not supported yet.
- A D3D12 or Vulkan game that uses DLSS Ray Reconstruction.
- `nvngx_dlssd.dll` 310.9.1. OpenRR builds its weights from this file when the game starts. If the game ships another
  version, Ray Reconstruction stays on FSR-RR.

## Install

Use the [OpenNR Installer](https://github.com/zmodelerlover/OpenNR-Installer/releases). On the OptiScaler route,
OpenRR is ticked by default on RX 9000. The installer puts `OpenNrRrRuntime.dll` and `opennr-rr\shaders\` next to
OptiScaler, and adds NVIDIA's 310.9.1 DLL when the game doesn't have it. On RX 7000 and RDNA3 / RDNA 3.5 graphics it is
experimental and off unless ticked; those cards also get `opennr-rr\shaders-gfx11\`.

In game, OpenRR settings are in OptiScaler's `[OpenNR-RR]` section and in the upscaler list of the Insert menu.

## How it runs

- **D3D12 games**: the kernels run on the game's own device and are recorded into the game's command list.
- **Vulkan games**: the kernels run on the game's `VkDevice` and are recorded into its command buffer.
- The network is built on a background thread at startup. Until it is ready, frames pass through scaled.
- **NR**: in D3D12 games, mochizuki's DLSS 5 NR can run on the denoised image; its change is kept in a history of its
  own and added to the displayed image, so the network's own history stays clean. In Vulkan games NR on OpenRR doesn't
  run yet.

Three modes, picked in the menu or with `Mode` in `[OpenNR-RR]` (`new`, `fast`, `legacy`), which flip live:

| | Performance 1080p | Quality 1080p | Against NVIDIA's own (RTX card) |
|---|---|---|---|
| **New** (default): cheaper kernels | about 2.0 ms | about 2.9 ms | about 60.8 dB on the display |
| **Fast (experimental)**: New with more cut, and the network's deepest levels read from the previous frame | about 1.8 ms | about 2.75 ms | about 55 dB on the display |
| **Legacy**: the kernels closest to NVIDIA's | about 2.2 ms | about 3.27 ms | about 63 dB on the display |

Measured on an RX 9070 XT, outside a game. The first start builds the kernels of New and Fast in the background, with the
image on Legacy until they're ready. Fast changes more from frame to frame in motion and on very bright points: look at
those before keeping it.

Tested in Cyberpunk 2077, Resident Evil Requiem, Hogwarts Legacy and Neverness to Everness (D3D12), and Indiana Jones
and the Great Circle (Vulkan). DOOM: The Dark Ages is not supported: its menu hides DLSS on AMD GPUs.

## Clean room

- The kernels, the weight loader and the runtime are our own work. No GPL Ray Reconstruction port was opened or
  used, and no OptiScaler code is copied here.
- What the network does was worked out from captures of NVIDIA's runtime on NVIDIA hardware, and every stage is
  checked against those captures.
- No NVIDIA binaries, weights or extracted kernels are in this repository or its releases. The weights are read from
  the player's own `nvngx_dlssd.dll` on their PC.

## Building

Needs Visual Studio 2026 (MSVC), CMake 3.24 or newer and the Vulkan SDK (for `glslangValidator`).

```
cmake -S gpu -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release --target runtime_stage
```

`build\stage\` then holds what ships: `OpenNrRrRuntime.dll` and `opennr-rr\shaders\`. The C interface is in
`runtime\include\opennr_rr_api.h`.

The RX 7000 kernels come from a second build, whose `stage\opennr-rr\shaders\` ships as `opennr-rr\shaders-gfx11\`:

```
cmake -S gpu -B build-gfx11 -G "Visual Studio 18 2026" -A x64 -DONR_GFX11=ON -DONR_SPLIT_C=128
cmake --build build-gfx11 --config Release --target runtime_stage
```

## Repository layout

| Path | What |
|---|---|
| `gpu\` | the kernels (GLSL compute), the engine, the weight loader and a test harness |
| `runtime\` | `OpenNrRrRuntime.dll`, its C interface and the D3D12 and Vulkan validators |

## License

MIT, see `LICENSE`. NVIDIA, DLSS and the DLSS model are NVIDIA's. This project is not affiliated with NVIDIA or AMD.
