# Compares Polyflow's tangent-space normal map with Blender's.
#
#   blender --background --factory-startup --python compare.py -- <dir>
#
# Expects in <dir>: blender_normal.npy (from make_pair.py), and ours_normal.png
# and ours_object.png from `polyflow-cli bake high.obj low.obj <dir>/ours`.
# Prints the angle between the two maps' normals over the texels Polyflow
# baked, next to the error 8-bit rounding alone causes, and writes error.png
# (black: equal, white: 10 degrees or more).
import os
import sys

import bpy
import numpy as np

d = sys.argv[sys.argv.index("--") + 1]


def load_png(name):
    img = bpy.data.images.load(os.path.join(d, name))
    img.colorspace_settings.name = "Non-Color"
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    return px.reshape(h, w, 4)


def decode(c):
    n = c[..., :3] * 2.0 - 1.0
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)


def angles(a, b):
    return np.degrees(np.arccos(np.clip((decode(a) * decode(b)).sum(-1), -1.0, 1.0)))


blender = np.load(os.path.join(d, "blender_normal.npy"))
ours = load_png("ours_normal.png")
# Polyflow's object-space map is exactly mid-gray outside the UV layout.
mask = np.abs(load_png("ours_object.png")[..., :3] - 128 / 255).max(-1) > 1e-6

error = angles(blender, ours)
rounding = angles(blender, np.round(blender * 255) / 255)
print(f"texels compared: {mask.sum()}")
for name, e in (("Polyflow vs Blender", error[mask]), ("8-bit rounding alone", rounding[mask])):
    print(
        f"{name:<21} mean {e.mean():.3f} deg, p50 {np.percentile(e, 50):.3f}, "
        f"p99 {np.percentile(e, 99):.3f}, p99.9 {np.percentile(e, 99.9):.3f}, max {e.max():.2f}"
    )
print(f"texels over 5 deg: {(error[mask] > 5).sum()}")

shade = np.clip(np.where(mask, error, 0) / 10.0, 0, 1).astype(np.float32)
out = bpy.data.images.new("error", shade.shape[1], shade.shape[0])
out.pixels.foreach_set(np.stack([shade, shade, shade, np.ones_like(shade)], -1).ravel())
out.filepath_raw = os.path.join(d, "error.png")
out.file_format = "PNG"
out.save()
print("wrote", out.filepath_raw)
