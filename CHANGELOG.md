# Changelog

All notable changes to Polyflow are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Selection**: the new **Select** tool (S) picks vertices, edges or faces
  (1, 2, 3) by clicking or by dragging a box or a lasso; Shift adds and Ctrl
  removes. Double-click selects the edge loop under the cursor, or in face
  mode the face loop. L adds the loops through the selected edges, Shift+L
  their rings; A selects or deselects everything, Esc deselects, and the
  panel can also invert the selection. Hidden elements are left out unless
  X-ray is on. The selection stays across tools, edits and undo.
- **Topology operations** on the selection, from keys or the panel: delete
  (Delete), dissolve (Backspace: vertices, edges and faces go and the faces
  around them merge), merge vertices at their center (M), connect two
  vertices across their face (J) and rotate edges (T).
- **Loop** tool (I): inserts an edge loop across a ring of quads, previewed
  under the cursor; drag along the edge to slide it, hold Shift for the
  middle.
- **Cut** tool (C): click on edges one after another to cut across faces.
- Dropping a vertex on another while moving it (Ctrl+drag in Create) welds
  them, in the same undo step as the move.
- All of them keep UVs (new corners interpolate them) and follow symmetry.

### Changed

- Delete removes the selection when there is one. Backspace now dissolves
  the selection instead of deleting what is under the cursor.

## [0.2.0] - 2026-10-24

Baking: bake normal maps and ambient occlusion from the sculpt onto the
retopology, on the GPU, from the app or the command line.

### Added

- **Bake panel** (right side; Ctrl+B to bake): map size, antialiasing,
  tangent-space normals (OpenGL or DirectX green), object-space normals and
  ambient occlusion, ray distances or a cage, edge padding, file format and
  output name. Bakes run in the background with a progress bar and Cancel;
  the settings are saved with the project.
- **Bake preview**: when a bake finishes, the viewport shows the low-poly as
  baked, lit with its normal map and occlusion, or the raw normal or
  occlusion map; the sculpt can be shown alongside or hidden.
- **GPU baking** with hardware ray tracing (`VK_KHR_ray_query`), or through
  the sculpt's BVH on GPUs without it, or on the CPU. All three give the
  same maps. A 4096x4096 bake of normal maps and AO from a 10M-triangle
  sculpt takes under a second on an RTX 4070 (about 10 s on a 32-thread
  CPU).
- Normal maps match what game engines expect: tangents are MikkTSpace,
  identical bit for bit to the reference implementation, and the bake writes
  `<name>_low.obj`, the low-poly triangulated with its normals exactly as
  baked, to use with the maps. Against Blender's Cycles bake of the same
  meshes, normals agree to 0.09° on average (`tools/validate-bake`).
- **Bake cage**: rays can start from a cage around the retopology, automatic
  (pushed out along the normals, adjustable) or edited in other software:
  **File → Export bake cage OBJ**, move vertices, **Import bake cage OBJ**.
  The cage can be shown in the viewport and is saved in the project.
- Maps as 8- or 16-bit PNG or as OpenEXR (half or full float), antialiased
  with up to 8x8 samples per texel, with edge padding so seams do not show
  with filtering and mipmaps.
- **UVs** on the retopology, per face corner: imported from OBJ, kept through
  edits and undo, written on export and saved in projects. A **UV checker**
  view shows them (faces without UVs in magenta: they cannot be baked).
- `polyflow-cli bake <sculpt> <retopo> <out>` with all the panel's options,
  and `polyflow-cli raster` to check a UV layout's coverage and overlaps.
- Importing a mesh of more than 500,000 faces as retopology offers to load it
  as the reference mesh instead.

### Fixed

- Windows: dialogs (unsaved changes on close, recovery, file pickers) could
  take seconds to appear or hang the app, with one CPU core at 100%. They now
  open on their own thread and show at once; the window keeps drawing and
  ignores input while one is open.

### Known limitations

- The retopology has no UV unwrapping: unwrap it in another program and
  import it. Faces without UVs are left out of the bake.
- Maps: tangent- and object-space normals and AO; height, cavity, curvature
  and ID maps come later. Normals are smooth (no hard edges yet).
- The baked low-poly and the maps' preview reflect the retopology at the time
  of the bake.
- Reference meshes: OBJ only.

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

[0.2.0]: https://github.com/DanielStivenRivera/polyflow/releases/tag/v0.2.0
[0.1.1]: https://github.com/DanielStivenRivera/polyflow/releases/tag/v0.1.1
[0.1.0]: https://github.com/DanielStivenRivera/polyflow/releases/tag/v0.1.0
