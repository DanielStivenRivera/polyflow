# Polyflow — Design Document

> Status: v0.1 released · Last updated: 2026-10-18

## 1. Vision

Polyflow is an open-source, industry-grade **retopology and baking** tool, in the
spirit of TopoGun. An artist loads a high-resolution sculpt (millions to hundreds
of millions of triangles), builds a clean, animation-ready low-poly mesh on top of
it, and bakes the surface detail of the sculpt into texture maps for the low-poly
mesh.

The guiding principle: **it must stay fast and stable on the heaviest meshes
production throws at it.** Performance on huge data is a feature, not an
optimization pass.

### Goals

- Interactive viewport for 50M+ triangle reference meshes on mid-range GPUs.
- A complete, ergonomic set of retopology tools (TopoGun-class).
- GPU-accelerated baking that matches what game engines expect (MikkTSpace).
- First-class pipeline integration: standard formats, headless CLI, scripting.
- Never lose the artist's work.

### Non-goals (for now)

- Sculpting, modeling from scratch, texture painting, animation.
- macOS support (left to the community; see §4).
- Automatic UV unwrapping (UVs are imported from a DCC tool in the first versions).

## 2. Core workflow

```
 Import high-poly ──► Retopologize ──► Export low-poly ──► (UVs in DCC) ──► Import UVs ──► Bake ──► Export maps
```

## 3. Feature scope

### Retopology

| Tool | Description |
|---|---|
| Create | Place vertices on the surface, build faces from them |
| Draw / Strips | Stroke-based generation of quad strips |
| Bridge / Fill | Connect edge loops, fill holes |
| Extrude | Extend boundary edges along the surface |
| Brush | Move / relax / smooth vertices while staying on the surface |
| Topology ops | Merge, split, insert loop, dissolve, flip, delete |
| Symmetry | Mirror on X/Y/Z with seam welding |
| Snapping | Continuous projection of the low-poly onto the reference |
| Undo / Redo | Unlimited, designed in from day one |

### Baking

- Tangent-space normals (MikkTSpace), object-space normals
- Ambient occlusion, height/displacement, cavity, curvature
- Vertex color / ID transfer
- Cage or ray-distance projection

### I/O

- Import: OBJ, PLY (v0.1); FBX, glTF, USD (later)
- Export mesh: OBJ (v0.1); FBX, glTF, USD (later)
- Export maps: PNG, TGA, EXR
- Native project format (versioned) and a fast binary cache for reference meshes

## 4. Platform and technology decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language | Rust (edition 2024) | Performance with memory safety |
| Platforms | Windows, Linux (X11 + Wayland) | Covers the vast majority of studios. macOS ports welcome from the community |
| Graphics API | Vulkan 1.3 via `ash` | Hardware ray tracing (`VK_KHR_ray_query`), mesh shaders, bindless, buffer device address, full control over memory and sync |
| GPU memory | `gpu-allocator` | Mature sub-allocator |
| Shaders | Slang → SPIR-V | Modern language, modules, generics; Khronos-hosted |
| UI | `egui` with a custom Vulkan renderer | Immediate mode, fast iteration |
| Windowing | `winit` | Cross-platform, X11 + Wayland |
| Math | `glam` | De facto standard |
| CPU parallelism | `rayon` | Embarrassingly parallel workloads (bake, BVH build, import) |
| License | MIT OR Apache-2.0 | Lowest friction for studio adoption |

### Vulkan feature tiers

Vulkan 1.3 core (dynamic rendering, synchronization2) is the **baseline**.
Advanced features are optional and always have a fallback:

| Feature | Used for | Fallback |
|---|---|---|
| `VK_KHR_ray_query` | GPU baking | BVH traversal in a compute shader |
| `VK_EXT_mesh_shader` | Meshlet rendering | Compute culling + multi-draw indirect |

## 5. Architecture

### Two meshes, two data models

- **Reference mesh (high-poly)**: immutable. Stored as compact, GPU-friendly
  buffers, split into meshlets (~128 triangles) for culling, plus a BVH for
  spatial queries. Optimized for size and read throughput.
- **Retopo mesh (low-poly)**: mutable. Stored as a half-edge structure with
  stable handles, optimized for topological editing. Typically < 100k faces.

### Crate layout

```
crates/
├─ core     # math, half-edge mesh, commands / undo. No GPU.
├─ spatial  # BVH, raycast, closest-point queries
├─ io       # importers, exporters, binary cache
├─ bake     # baking (CPU reference + GPU)
├─ render   # Vulkan renderer (ash)
├─ app      # egui UI, tools, input, viewport  → binary `polyflow`
└─ cli      # headless batch baking/conversion  → binary `polyflow-cli`
```

Dependency rule: `core` depends on nothing internal; `render` and `app` are the
only crates allowed to touch the GPU or the window. Everything except `render`
and `app` must build and test on a headless CI machine.

### Snapping vs. baking

- **Snapping** runs on the CPU BVH: few rays, latency-critical (< 1 ms).
- **Baking** runs on the GPU: millions of rays, throughput-critical.

## 6. Performance budgets

Reference hardware: mid-range desktop GPU (e.g. RTX 3060 / RX 6600 class),
8-core CPU, NVMe SSD. Reference meshes: Stanford Lucy (28M tris), Thai Statue
(10M), XYZ RGB Dragon (7.2M).

| Metric | Budget |
|---|---|
| Import 10M-triangle OBJ | < 3 s |
| Reopen 10M triangles from cache | < 0.5 s |
| Viewport, 50M triangles | ≥ 60 fps |
| Snap query (single point) | < 1 ms |
| Retopo edit operation | < 16 ms (one frame) |
| Normal + AO bake, 4K, 10M ref, RT GPU | < 10 s |
| Reference mesh memory | ≤ 32 bytes / triangle (CPU + GPU, excl. BVH) |

Budgets are tracked with `criterion` benchmarks; regressions fail review.
Profiling uses Tracy (CPU + GPU zones).

## 7. Quality and robustness

- Vulkan validation layers enabled in debug builds.
- Half-edge invariants checked after every operation in debug builds; property
  tests with `proptest`.
- Importers fuzzed with `cargo-fuzz` — malformed files must never crash.
- Bake output verified against reference images; tangent space must match
  MikkTSpace exactly.
- Autosave, panic-time recovery snapshot, versioned project format.

## 8. Roadmap

| Version | Content |
|---|---|
| **v0.1** ✅ | Fast viewer for huge meshes · basic retopo (create, strips, relax, symmetry) · undo/redo · OBJ import/export |
| **v0.2** | GPU baking (normal, AO) · UV import · cage · PNG/EXR export |
| **v0.3** | Full retopo toolset · FBX / glTF · project files · autosave |
| **v0.4** | Remaining bake maps · CLI batch baking · Python bindings (`pyo3`) |
| **v1.0** | USD · DCC bridges (Blender, Maya, ZBrush) · stability and polish |

### v0.1 milestones

1. Window, Vulkan device and swapchain, egui overlay, orbit camera.
2. OBJ import (parallel parser) and naive rendering of the reference mesh.
3. Meshlet build and GPU culling; hit the 50M-triangle budget.
4. CPU BVH, raycast and closest-point queries.
5. Half-edge mesh, command system, undo/redo.
6. Create, strips, relax and symmetry tools.
7. OBJ export, first public release.

## 9. Open questions

- Pen tablet pressure input: `winit` support is limited; may need native
  Windows Ink / libinput handling.
- Project file format: custom binary vs. a container such as SQLite.
- UV unwrapping: in scope after v1.0?
