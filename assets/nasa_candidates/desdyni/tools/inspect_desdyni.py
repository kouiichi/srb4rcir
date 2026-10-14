"""Inspect and render the unmodified NASA candidate in source coordinates."""
import argparse
import json
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--blend", required=True)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(Path(args.source).resolve()))
scene = bpy.context.scene
scene.unit_settings.system = "METRIC"
scene.unit_settings.scale_length = 1.0
meshes = [obj for obj in scene.objects if obj.type == "MESH"]


def bounds(points):
    return [[min(p[i] for p in points) for i in range(3)],
            [max(p[i] for p in points) for i in range(3)]]


records = []
all_points = []
for obj in meshes:
    points = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    all_points.extend(points)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-5)
    seen = set()
    components = []
    for vertex in bm.verts:
        if vertex in seen:
            continue
        stack = [vertex]
        seen.add(vertex)
        connected = []
        while stack:
            current = stack.pop()
            connected.append(obj.matrix_world @ current.co)
            for edge in current.link_edges:
                other = edge.other_vert(current)
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        components.append({"vertices": len(connected), "bounds": bounds(connected)})
    records.append({"name": obj.name, "bounds": bounds(points),
                    "vertices": len(obj.data.vertices), "triangles": len(obj.data.polygons),
                    "welded_boundary_edges": sum(edge.is_boundary for edge in bm.edges),
                    "welded_nonmanifold_edges": sum(not edge.is_manifold for edge in bm.edges),
                    "connected_components": sorted(components, key=lambda c: c["vertices"], reverse=True)[:24]})
    bm.free()
audit = {"coordinate_units": "source coordinates; physical scale uncalibrated",
         "world_bounds": bounds(all_points), "objects": records,
         "component_weld_tolerance_source_units": 1e-5}
(output / "geometry_inspection.json").write_text(json.dumps(audit, indent=2) + "\n")
bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.blend).resolve()))

scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 12
scene.cycles.use_denoising = True
scene.render.resolution_x = 1100
scene.render.resolution_y = 900
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
world = bpy.data.worlds.new("InspectionWorld")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (.25, .29, .34, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = .7
for index, location in enumerate(((12, -15, 18), (-12, -8, -12), (0, 18, 10))):
    data = bpy.data.lights.new(f"InspectionLight{index}", "AREA")
    data.energy = 4500
    data.shape = "DISK"
    data.size = 12
    light = bpy.data.objects.new(data.name, data)
    scene.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (-light.location).to_track_quat("-Z", "Y").to_euler()
camera_data = bpy.data.cameras.new("InspectionCamera")
camera_data.type = "ORTHO"
camera = bpy.data.objects.new("InspectionCamera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
views = (("whole_oblique", (22, -30, 18), (0, 0, 0), 26),
         ("whole_front_minus_x", (-35, 0, 0), (0, 0, 0), 26),
         ("bottom_minus_z", (0, 0, -30), (0, 0, 0), 26),
         ("bus_oblique", (10, -12, -9), (0, 0, 0), 8))
for name, location, target, ortho_scale in views:
    camera.location = location
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.ortho_scale = ortho_scale
    scene.render.filepath = str(output / f"{name}.png")
    bpy.ops.render.render(write_still=True)
print("INSPECTION_COMPLETE", json.dumps({"bounds": audit["world_bounds"], "mesh_objects": len(records)}))
