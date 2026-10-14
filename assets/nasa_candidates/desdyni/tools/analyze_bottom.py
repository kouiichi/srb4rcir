"""Measure the candidate bottom surface and render its operation region."""
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()
hits = []
for x, y in ((0, 0), (.5, 0), (1, 0), (1.5, 0), (1.8, 0), (2, 0),
             (0, .5), (0, 1), (0, 1.5), (0, 1.8), (0, 2)):
    hit, location, normal, face_index, obj, matrix = scene.ray_cast(
        depsgraph, Vector((x, y, -20)), Vector((0, 0, 1)))
    entry = {"x": x, "y": y, "hit": hit}
    if hit:
        evaluated = obj.evaluated_get(depsgraph)
        material_index = evaluated.data.polygons[face_index].material_index
        material = evaluated.data.materials[material_index]
        entry.update(position=list(location), normal=list(normal), object=obj.name,
                     face_index=face_index, material=material.name if material else None)
    hits.append(entry)
groups = []
for obj in scene.objects:
    if obj.type != "MESH":
        continue
    for index, material in enumerate(obj.data.materials):
        polygons = [p for p in obj.data.polygons if p.material_index == index]
        points = [obj.matrix_world @ obj.data.vertices[v].co for p in polygons for v in p.vertices]
        if not points:
            continue
        lower = [min(p[i] for p in points) for i in range(3)]
        upper = [max(p[i] for p in points) for i in range(3)]
        if lower[2] < -3:
            groups.append({"object": obj.name, "material_slot": index,
                           "material": material.name if material else None,
                           "triangles": len(polygons), "bounds": [lower, upper]})
body = bpy.data.objects["Layer_0"]
body_points = [body.matrix_world @ v.co for v in body.data.vertices]
bottom_z = min(p.z for p in body_points)
rim_points = sorted(set((round(p.x, 6), round(p.y, 6)) for p in body_points
                        if abs(p.z - bottom_z) < 5e-4))


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


lower, upper = [], []
for p in rim_points:
    while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
        lower.pop()
    lower.append(p)
for p in reversed(rim_points):
    while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
        upper.pop()
    upper.append(p)
rim_hull = lower[:-1] + upper[:-1]
record = {"bottom_direction_source": [0, 0, -1], "axial_ray_samples": hits,
          "lower_material_groups": groups, "physical_scale_calibrated": False,
          "bottom_rim_candidate": {"z": bottom_z, "selection_tolerance": 5e-4,
                                   "unique_xy_points": len(rim_points),
                                   "convex_hull_xy": rim_hull,
                                   "method": "projected convex hull of Layer_0 minimum-Z vertices; requires visual contour review"}}
(output / "bottom_inspection.json").write_text(json.dumps(record, indent=2) + "\n")
if "--measure-only" in sys.argv:
    print("RIM_MEASUREMENT", json.dumps(record["bottom_rim_candidate"]))
    sys.exit(0)
world = bpy.data.worlds.new("InspectionWorld")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (.25, .29, .34, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = .8
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 12
scene.cycles.use_denoising = True
scene.render.resolution_x = 1100
scene.render.resolution_y = 900
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
for index, location in enumerate(((10, -12, -15), (-10, 8, -12), (0, 2, 12))):
    data = bpy.data.lights.new(f"InspectionLight{index}", "AREA")
    data.energy = 4000
    data.size = 10
    light = bpy.data.objects.new(data.name, data)
    scene.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (Vector((0, 0, -4)) - light.location).to_track_quat("-Z", "Y").to_euler()
camera_data = bpy.data.cameras.new("InspectionCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 5.5
camera = bpy.data.objects.new("InspectionCamera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
for name, location in (("bottom_close_original", (0, 0, -20)),
                       ("bottom_oblique_original", (9, -11, -16))):
    camera.location = location
    camera.rotation_euler = (Vector((0, 0, -4)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(output / f"{name}.png")
    bpy.ops.render.render(write_still=True)
print("BOTTOM_INSPECTION_COMPLETE", json.dumps({"ray_samples": hits}))
