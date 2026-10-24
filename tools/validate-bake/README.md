# Bake validation against Blender

Checks that Polyflow's tangent-space normal maps match what Cycles bakes
from the same meshes. Both bakers use MikkTSpace and Blender's conventions
(shortest-diagonal triangulation, angle-weighted smooth normals), so the
maps should agree to within 8-bit rounding.

Needs Blender 4.2 or later (its bundled Python has numpy).

```sh
blender --background --factory-startup --python tools/validate-bake/make_pair.py -- out 1024 0.05
polyflow-cli bake out/high.obj out/low.obj out/ours --size 1024 --distance 0.05 --padding 0 --format exr32
blender --background --factory-startup --python tools/validate-bake/compare.py -- out
```

The bake runs on the GPU if there is one; `--cpu` checks the CPU baker,
which gives the same figures. `--padding 0` matters: the comparison finds the baked texels by the
background Polyflow leaves around the UV islands. Without `--format exr32`
the maps are compared as 8-bit PNGs.

Result with Blender 5.0.1 and Polyflow 0.2-dev (722,576 texels):

| | mean | p50 | p99 | p99.9 |
|---|---|---|---|---|
| Polyflow (32-bit float) vs Blender | 0.087° | 0.052° | 0.54° | 0.97° |
| Polyflow (8-bit PNG) vs Blender | 0.203° | 0.192° | 0.61° | 1.02° |
| 8-bit rounding of Blender's map alone | 0.171° | 0.178° | 0.30° | 0.32° |

6 texels differ by more than 5°, where rays graze folds of the sculpt.
Leaving the high-poly as quads raises the mean to about 0.3°: the two
programs split its non-planar quads along different diagonals.
