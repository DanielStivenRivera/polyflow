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
and `app` must build and test on a headless CI machine. `render` uses `bake`
and `spatial` for the GPU baker, which shares everything but ray tracing with
the CPU one; `cli` bakes on the GPU through `render` without a window.

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
| **v0.2** ✅ | GPU baking (normal, AO) · UV import · cage · PNG/EXR export |
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

### v0.2 milestones

UVs are unwrapped in a DCC and imported; unwrapping inside Polyflow stays out
of scope. A CPU baker comes first and is the reference the GPU baker is
validated against.

1. UVs on the retopology: per-corner UVs in the half-edge mesh, kept through
   edits and undo; `vt` in OBJ import/export; UVs in `.pflow`; checker
   preview in the viewport.
2. ✅ Texture-space rasterization: triangulate the low-poly, per-texel
   position, normal and tangent; tangent basis matching MikkTSpace exactly.
3. ✅ CPU reference baker: tangent- and object-space normals and AO with
   ray-distance projection, on the CPU BVH with `rayon`; analytic tests, and
   a cross-check against Cycles (`tools/validate-bake`).
4. ✅ Cage: automatic (low-poly pushed along averaged normals, adjustable) or
   imported from OBJ; shown in the viewport.
5. ✅ Map export: PNG 8/16-bit and EXR half/float, with edge padding
   (dilation) and supersampling.
6. ✅ GPU baker: Vulkan compute with `VK_KHR_ray_query`, compute BVH
   traversal as fallback; matches the CPU reference within tolerance; hits
   the bake budget.
7. ✅ Bake panel with progress and cancel, map preview on the low-poly,
   `polyflow-cli bake`, export of the low-poly as baked (see below), release
   v0.2.0.

### The low-poly as baked

A tangent-space normal map is only correct for the exact triangles, normals
and tangents it was baked with, and engines rebuild tangents from the mesh
they import. Polyflow fixes all three (`polyflow-bake::mesh`):

- **Triangles**: quads split along their shorter diagonal, other polygons
  fanned from their first corner.
- **Normals**: smooth, angle-weighted vertex normals computed on the polygons
  (independent of the triangulation). Hard edges come later.
- **Tangents**: MikkTSpace through `bevy_mikktspace`, which matches the
  reference C code bit for bit; a test compares against that C code
  (`mikktspace-sys`). The encoding inverts a MikkTSpace shader exactly: `N`
  and `T` interpolated unnormalized, `B = sign · cross(N, T)` per pixel.

So the mesh sent to the engine must be exported triangulated, with these
normals, as part of the bake output. These are also Blender's conventions:
`tools/validate-bake` bakes the same meshes in Cycles, and the normal maps
agree to within 8-bit rounding (mean 0.2°).

### Projection

The CPU baker projects by ray distance: each texel's ray starts a frontal
distance in front of the low-poly (along its smooth normal) and runs back to
a rear distance behind it; the first reference surface met is baked, as
Cycles and Substance do. Texels whose ray meets nothing are reported as
misses. Ambient occlusion traces cosine-weighted rays from the hit, using the
same Hammersley points in every texel, shifted per texel, so results are
deterministic whatever the thread count. UV rasterization uses fixed-point
coordinates (1/256 texel) and the top-left rule, like a GPU, so every texel
inside a UV layout is covered exactly once.

With a cage, the ray instead starts at the texel's point on the cage and
runs through its point on the low-poly. A cage is the low-poly with moved
vertices (`polyflow-bake::cage`): automatic (pushed along the smooth
normals, so it stays closed across UV seams; on a flat surface it casts
exactly the rays of the frontal distance) or imported. An imported cage is
matched to the retopology by vertex order and checked face by face, so the
workflow is: export the cage from Polyflow, edit it elsewhere moving
vertices only, import it back. It is stored in the project as `cage.obj`
and falls back to the automatic cage once vertices are added or removed.

### Output

Supersampling takes `n`² samples per texel on a regular grid, one
rasterization pass per sample position; maps average the samples that hit
(normals as vectors, renormalized). Sums accumulate in the output buffers
themselves, so it costs two counters per texel, not a buffer per sample.
Edge padding then grows the islands ring by ring, each new texel averaging
its already-filled neighbors (copying the nearest texel instead smears noisy
border texels into streaks). Maps are written as 8- or 16-bit PNG or as
half/float OpenEXR, as linear data with no color space conversion.

### GPU baking

`polyflow-render::bake` runs the CPU baker's per-sample work in a compute
shader, a line-by-line port: same projection, tangent encoding, occlusion
points and per-texel hash. Rays are traced with `VK_KHR_ray_query` against
an acceleration structure of the reference (one opaque, two-sided geometry)
or, on GPUs without it, through the reference's own BVH (`Bvh::gpu_nodes`)
with the CPU's traversal and triangle test. The CPU still rasterizes the UVs
and finishes the maps (`polyflow_bake::finish`), so only the ray tracing
moves. The map is processed in bands of rows sized to about 32M rays per
dispatch, so no dispatch runs long enough for the OS to reset the GPU, and
per-band sums stay on the GPU until read back.

Both GPU paths agree with the CPU baker to 0.0006° on average (AO and
coverage identical) in `gpu_matches_cpu`. On an RTX 4070, a 4096² bake of
both normal maps and AO (32 rays) from a 10M-triangle sculpt takes 0.76 s
with ray queries, 2.2 s with the compute fallback and 9.5 s on a 32-thread
CPU; with 4x4 samples per texel, 2.7 s.

## 9. Open questions

- Pen tablet pressure input: `winit` support is limited; may need native
  Windows Ink / libinput handling.
- ~~Project file format~~: decided in v0.1.1 — a zip (`.pflow`) holding a
  versioned `project.json` and the retopology as polygon OBJ. Easy to inspect
  and to extend with new entries (UVs, baked maps) without breaking old files.
- UV unwrapping: in scope after v1.0?
