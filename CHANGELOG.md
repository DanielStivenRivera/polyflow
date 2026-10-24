# Changelog

All notable changes to Polyflow are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- UVs on the retopology, per face corner (seams need no split vertices).
  Imported from OBJ (`vt`), written back on export and kept in project
  files. Edits and undo keep them; faces built with the retopology tools
  have none.
- **UV checker** view: faces are shaded with a checker from their UVs, tinted
  red along U and green along V; faces without UVs show in magenta. Turned on
  automatically when an imported retopology has UVs. The panel shows how many
  faces have UVs.
- Importing a mesh of more than 500,000 faces as retopology offers to load it
  as the reference mesh instead.

### Fixed

- Windows: dialogs (unsaved changes on close, recovery, file pickers) could
  take seconds to appear or hang the app, with one CPU core at 100%. They now
  open on their own thread and show at once; the window keeps drawing and
  ignores input while one is open.

## [0.1.1] - 2026-10-19

### Added

- Project files (`.pflow`): the retopology, the sculpt's location (relative
  to the project and absolute, so moving a folder keeps working), the camera
  and the retopology and tool settings, in one file. File > New, Open, Save
  and Save as. If the sculpt has moved, Polyflow asks you to locate it.
- Autosave every 2 minutes while there are unsaved changes, and an offer to
  recover that work after Polyflow closes unexpectedly.
- Projects open by dropping them on the window or from the command line.

### Changed

- Shortcuts: Ctrl+O opens a project and Ctrl+S saves it; the reference mesh
  loads with Ctrl+L; retopology OBJ import and export are Ctrl+I and Ctrl+E.

### Fixed

- Windows: no console window opens next to the app (release builds). Logs and
  `--bench` output still appear when started from a terminal.

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
- No native project format: save the retopology as OBJ (added in 0.1.1).

[0.1.1]: https://github.com/DanielStivenRivera/polyflow/releases/tag/v0.1.1
[0.1.0]: https://github.com/DanielStivenRivera/polyflow/releases/tag/v0.1.0
