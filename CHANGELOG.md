# Changelog

All notable changes to Polyflow are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-18

First public preview: load a sculpt, retopologize it by hand, save the result.

### Added

- Vulkan 1.3 viewport for huge reference meshes, with GPU meshlet culling
  (frustum and backface). A 50M-triangle mesh renders in ~3 ms on an RTX 4070.
- Parallel OBJ import: a 10M-triangle, 410 MB file is ready to work on in
  about 1.3 s, including meshlets and the BVH. File > Open, drag and drop, or a path on the
  command line.
- Orbit camera with Blender- and Maya-style bindings, zoom to cursor and auto
  depth.
- Retopology tools: **Create** (place vertices, build quads and polygons, move,
  delete), **Draw** (strokes become connected quad strips) and **Relax** (a
  smoothing brush that keeps vertices on the surface).
- Mirror symmetry on X, Y or Z for every tool.
- Undo/redo (1000 steps); every action, drag or brush stroke is one step, and
  failed edits are rolled back.
- Save and import the retopology as polygon OBJ (quads stay quads), with a
  prompt before losing unsaved work.
- `polyflow-cli info` and `polyflow-cli bvh` for headless loading and
  benchmarks; a `--bench` mode in the app for reproducible GPU timings.

### Known limitations

- Reference meshes: OBJ only (no PLY, FBX, glTF or USD yet).
- No baking yet (planned for v0.2).
- Missing retopology tools: selection, edge loops, bridge, extrude.
- No native project format: save the retopology as OBJ.

[0.1.0]: https://github.com/DanielStivenRivera/polyflow/releases/tag/v0.1.0
