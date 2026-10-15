"""Check that front wells, mounting bores and rear tube openings remain open."""
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

folder = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
parameters = json.loads((folder / "reference_parameters.json").read_text())
scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()
results = {"front_valve_wells": [], "open_mount_holes": 0, "rear_tube_centers": []}
for x in (-.015, .015):
    hit, point, normal, face, obj, matrix = scene.ray_cast(
        depsgraph, Vector((x, 0, .1)), Vector((0, 0, -1)))
    assert hit and point.z < .048260-.003, (x, point, obj)
    results["front_valve_wells"].append({"x": x, "first_hit_z": point.z, "part": obj.name})
    hit, point, normal, face, obj, matrix = scene.ray_cast(
        depsgraph, Vector((x, 0, -.1)), Vector((0, 0, 1)))
    assert hit and point.z > -.001, (x, point, obj)
    results["rear_tube_centers"].append({"x": x, "first_hit_z": point.z, "part": obj.name})
for x, y in parameters["mount_hole_xy_m"]:
    hit, point, normal, face, obj, matrix = scene.ray_cast(
        depsgraph, Vector((x, y, .1)), Vector((0, 0, -1)))
    assert not hit, (x, y, point, obj)
    results["open_mount_holes"] += 1
(folder / "blender_geometry_audit.json").write_text(json.dumps(results, indent=2) + "\n")
print("GEOMETRY_AUDIT", json.dumps(results))
