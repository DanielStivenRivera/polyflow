# Architecture

This is a map of the codebase. For the reasoning behind the decisions, see
[docs/DESIGN.md](docs/DESIGN.md).

## Bird's-eye view

Polyflow works with two very different meshes:

- The **reference mesh** (the high-poly sculpt) is immutable and huge. It lives
  in compact GPU buffers split into meshlets, plus a BVH for spatial queries.
- The **retopo mesh** (the low-poly the artist builds) is small and constantly
  edited. It lives in a half-edge structure with stable handles.

Retopology tools edit the retopo mesh and use the BVH to keep it snapped to the
reference surface. Baking shoots rays from the retopo mesh to the reference
mesh and writes the results into textures.

## Crates

```
crates/
├─ core     polyflow-core     math, half-edge mesh, commands / undo
├─ spatial  polyflow-spatial  BVH, raycast, closest-point queries
├─ io       polyflow-io       importers, exporters, binary cache
├─ bake     polyflow-bake     baking (CPU reference + GPU)
├─ render   polyflow-render   Vulkan renderer (ash)
├─ app      polyflow          UI, tools, viewport      → binary `polyflow`
└─ cli      polyflow-cli      headless batch operations → binary `polyflow-cli`
```

```
        app ──► render ──┐
         │               ▼
         └──► io, bake ─► spatial ─► core
   cli ──────► io, bake ─┘
```

## Editing model

The retopo mesh is a `polyflow_core::halfedge::HalfEdgeMesh`: polygons with
half-edge connectivity, stored in slot arrays so element ids stay stable across
removals and undo. Topological operations (`add_face`, `remove_face`, ...)
validate their input before changing anything and refuse anything that would
make the mesh non-manifold.

Edits run through `polyflow_core::history::MeshDocument::edit`. While an edit
runs, the mesh journals the before/after value of every element it writes; a
failed (or panicking) edit is rolled back from that journal, and a successful
one becomes an undo step. Undo support therefore comes for free with any new
operation, as long as it only writes through the mesh's journaled setters.

## Invariants

- `core`, `spatial`, `io` and `bake` never touch the GPU or a window, and
  `#![forbid(unsafe_code)]`. They must build and test on headless CI.
- `render` is the only place with `unsafe` Vulkan calls. Every `unsafe` block
  carries a `// SAFETY:` comment.
- Importers treat all input as untrusted: malformed files return errors, never
  panic.
- Every edit to the retopo mesh goes through the command system, so it can be
  undone.
