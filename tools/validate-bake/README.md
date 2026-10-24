# Bake validation against Blender

Checks that Polyflow's tangent-space normal maps match what Cycles bakes
from the same meshes. Both bakers use MikkTSpace and Blender's conventions
(shortest-diagonal triangulation, angle-weighted smooth normals), so the
maps should agree to within 8-bit rounding.

Needs Blender 4.2 or later (its bundled Python has numpy).

```sh
blender --background --factory-startup --python tools/validate-bake/make_pair.py -- out 1024 0.05
polyflow-cli bake out/high.obj out/low.obj out/ours --size 1024 --distance 0.05
blender --background --factory-startup --python tools/validate-bake/compare.py -- out
```

Result with Blender 5.0.1 and Polyflow 0.2-dev (722,576 texels):

| | mean | p99 | p99.9 |
|---|---|---|---|
| Polyflow vs Blender | 0.203° | 0.61° | 1.02° |
| 8-bit rounding alone | 0.171° | 0.30° | 0.32° |

6 texels differ by more than 5°, where rays graze folds of the sculpt.
Leaving the high-poly as quads raises the mean to about 0.3°: the two
programs split its non-planar quads along different diagonals.
