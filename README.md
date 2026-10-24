# Polyflow

**Open-source retopology and baking for high-poly meshes.**

Polyflow lets artists build clean, production-ready low-poly meshes on top of
multi-million-triangle sculpts, and bake the sculpt's detail into texture maps
(normals, AO, height and more). It is built in Rust on Vulkan, with one rule
above all: stay fast and stable on the heaviest meshes production throws at it.

> **Status:** v0.2.0 preview. Retopology and baking (normal maps and ambient
> occlusion, on the GPU) work end to end. See the [changelog](CHANGELOG.md)
> and the [roadmap](docs/DESIGN.md#8-roadmap).

## Download

Prebuilt binaries for Windows and Linux are attached to each
[release](https://github.com/DanielStivenRivera/polyflow/releases). Unzip and
run `polyflow` (or `polyflow.exe`).

## Platforms

- Windows 10/11
- Linux (X11 and Wayland)

Requires a GPU and driver with Vulkan 1.3 (any desktop GPU from roughly 2016
on, with current drivers). On Linux, the Vulkan loader (`libvulkan1`) and an X11
or Wayland session are needed. macOS ports are welcome from the community.

## Building

```sh
cargo build --release
cargo run --release
```

You need a stable Rust toolchain (installed automatically from
`rust-toolchain.toml` via rustup) and a C compiler (for the MikkTSpace
reference used in tests). The [Vulkan SDK](https://vulkan.lunarg.com/) is
optional: it provides validation layers for debug builds and `slangc` to
rebuild shaders, which otherwise come precompiled.

## Usage

Load a reference mesh (`.obj` for now) with **File → Load reference mesh**
(Ctrl+L), by dropping it on the window, or from the command line:

```sh
polyflow path/to/sculpt.obj
```

Navigation follows Blender or Maya conventions: orbit with MMB or Alt+LMB, pan
with Shift+MMB or Alt+MMB, zoom with the wheel, Ctrl+MMB or Alt+RMB, and press
F to frame the mesh.

### Retopology

| Tool | Use |
|---|---|
| **Create** (Q) | Click the surface to place vertices, or click existing ones; four make a quad, clicking the first vertex closes any polygon. Ctrl+drag moves a vertex along the surface. Delete removes the vertex or face under the cursor. |
| **Draw** (D) | Drag across the surface to lay down a strip of quads. Strips snap to nearby vertices, so they connect. |
| **Relax** (R) | Drag to smooth the retopo mesh; vertices stay on the surface and borders stay pinned unless *Relax borders* is on. |

Symmetry (X toggles it on the X axis) mirrors every edit. Ctrl+Z / Ctrl+Y undo
and redo; Esc or a right click cancels the face in progress. If the retopo
mesh sinks under strongly curved areas, raise *Offset* in the Retopology panel.

Save your work as a project with **File → Save project** (Ctrl+S): a `.pflow`
file with the retopology, where the sculpt is, the camera and your settings.
**Open project** (Ctrl+O, or drop the file on the window) brings it all back.
Unsaved work is autosaved every 2 minutes and offered back if Polyflow closes
unexpectedly. To take the retopology to other software, use **Export
retopology OBJ** (Ctrl+E); quads stay quads.

The **Bake cage** checkbox shows the envelope bake rays start from: the
retopology pushed out by *push* (3% of the sculpt's size by default). To shape
it by hand, use **File → Export bake cage OBJ**, move its vertices in your
modeling software (do not add, delete or reorder any) and bring it back with
**Import bake cage OBJ**; it is saved with the project.

### Baking

The retopology needs UVs: unwrap it in your modeling software and import it
(**File → Import retopology OBJ**); the **UV checker** shows them. Then, in
the **Bake** panel on the right, pick the maps and press **Bake** (Ctrl+B).
Rays start a set distance in front of the retopology (or from the bake cage)
and look a set distance behind it; if the panel reports many missed texels,
raise those distances. When the bake finishes, the viewport previews the
low-poly with its maps.

Next to the maps, the bake writes `<name>_low.obj`: the low-poly triangulated
and with the normals it was baked with. Use that mesh in your engine with the
maps, so the normal map shades as baked.

The headless `polyflow-cli` loads meshes without a window, which is handy for
pipelines and for benchmarking:

```sh
polyflow-cli info path/to/sculpt.obj
polyflow-cli raster path/to/retopo.obj 4096   # prepare for baking, rasterize the UVs
polyflow-cli bake sculpt.obj retopo.obj out/character --size 4096 --samples 2 --format png16
polyflow-cli bake sculpt.obj retopo.obj out/character --cage character_cage.obj
```

Baking runs on the GPU when there is one (with hardware ray tracing where
available) and on the CPU otherwise, or with `--cpu`; both give the same maps.

## Documentation

- [Design document](docs/DESIGN.md) — goals, decisions, performance budgets, roadmap
- [Architecture](ARCHITECTURE.md) — how the code is organized
- [Contributing](CONTRIBUTING.md)

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
