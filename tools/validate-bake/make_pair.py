# Builds a low/high test pair and bakes a reference tangent-space normal map
# with Cycles, to check Polyflow's baker against Blender's.
#
#   blender --background --factory-startup --python make_pair.py -- <out_dir> [size] [distance]
#
# Writes low.obj (Suzanne, subdivided once, triangulated on the shortest
# diagonal), high.obj (Suzanne subdivided 4 more times and displaced by noise,
# triangulated) and blender_normal.npy (the baked map, float RGBA, rows from
# the bottom). Both meshes are triangulated so neither baker has to guess the
# other's triangulation. The bake casts rays from `distance` in front of the
# low-poly to `distance` behind it, like `polyflow-cli bake --distance`.
import os
import sys

import bpy
import numpy as np

args = sys.argv[sys.argv.index("--") + 1:]
out_dir = args[0]
size = int(args[1]) if len(args) > 1 else 1024
distance = float(args[2]) if len(args) > 2 else 0.05
os.makedirs(out_dir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)


def apply(obj, modifier):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def triangulate(obj):
    tri = obj.modifiers.new("tri", "TRIANGULATE")
    tri.quad_method = "SHORTEST_DIAGONAL"
    tri.ngon_method = "BEAUTY"
    apply(obj, tri)


bpy.ops.mesh.primitive_monkey_add(size=2.0)
low = bpy.context.active_object
low.name = "low"
sub = low.modifiers.new("sub", "SUBSURF")
sub.levels = 1
apply(low, sub)
bpy.ops.object.shade_smooth()

high = low.copy()
high.data = low.data.copy()
high.name = "high"
bpy.context.collection.objects.link(high)
sub = high.modifiers.new("sub", "SUBSURF")
sub.levels = sub.render_levels = 4
apply(high, sub)
noise = bpy.data.textures.new("noise", "CLOUDS")
noise.noise_scale = 0.12
disp = high.modifiers.new("disp", "DISPLACE")
disp.texture = noise
disp.strength = 0.03
apply(high, disp)
triangulate(high)
triangulate(low)


def export(obj, name):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.wm.obj_export(
        filepath=os.path.join(out_dir, name),
        export_selected_objects=True,
        export_uv=True,
        export_normals=False,
        export_materials=False,
        forward_axis="Y",
        up_axis="Z",
    )


export(low, "low.obj")
export(high, "high.obj")
print(f"low: {len(low.data.polygons)} triangles, high: {len(high.data.polygons)} triangles")

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 1
image = bpy.data.images.new("baked", size, size, alpha=True, float_buffer=True)
image.colorspace_settings.name = "Non-Color"
mat = bpy.data.materials.new("bake")
mat.use_nodes = True
node = mat.node_tree.nodes.new("ShaderNodeTexImage")
node.image = image
mat.node_tree.nodes.active = node
low.data.materials.append(mat)

bpy.ops.object.select_all(action="DESELECT")
high.select_set(True)
low.select_set(True)
bpy.context.view_layer.objects.active = low
bake = scene.render.bake
bake.use_selected_to_active = True
bake.cage_extrusion = distance
bake.max_ray_distance = 2 * distance
bake.margin = 0
bake.normal_space = "TANGENT"
bpy.ops.object.bake(type="NORMAL")

pixels = np.empty(size * size * 4, dtype=np.float32)
image.pixels.foreach_get(pixels)
np.save(os.path.join(out_dir, "blender_normal.npy"), pixels.reshape(size, size, 4))
print("wrote", os.path.join(out_dir, "blender_normal.npy"))
