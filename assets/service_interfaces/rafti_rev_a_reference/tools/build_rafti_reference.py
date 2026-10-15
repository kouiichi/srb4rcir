"""Build a detachable RAFTI Rev A visual reference from published envelopes."""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.unit_settings.system = "METRIC"
bpy.context.scene.unit_settings.scale_length = 1
bpy.context.preferences.filepaths.save_version = 0

W = .096012
H = .048260
REAR = .0516128
PORT_X = .015


def empty(name, parent=None, position=(0, 0, 0)):
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent
    obj.location = position
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = .005
    return obj


visuals = empty("Visuals")
frames = empty("Frames")
empty("Mount", frames)
empty("Mate", frames, (0, 0, H))
empty("FluidA", frames, (-PORT_X, 0, H))
empty("FluidB", frames, (PORT_X, 0, H))


def material(name, color, metallic, roughness):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    return mat


steel = material("MachinedSteel", (.48, .51, .54), .85, .27)
dark_steel = material("HousingSteel", (.29, .31, .33), .85, .32)
polished = material("ValveTrim", (.65, .68, .71), .92, .19)
black = material("ValveRecess", (.012, .015, .019), .2, .48)
letter = material("FaceMarkings", (.018, .025, .033), .15, .4)


def mesh_object(name, vertices, faces, mat):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = visuals
    obj.data.materials.append(mat)
    return obj


def prism(name, outline, z0, z1, mat):
    n = len(outline)
    points = [(x, y, z) for z in (z0, z1) for x, y in outline]
    faces = [tuple(reversed(range(n))), tuple(range(n, 2*n))]
    faces += [(i, (i+1) % n, (i+1) % n+n, i+n) for i in range(n)]
    return mesh_object(name, points, faces, mat)


def annulus(name, x, y, inner, outer, z0, z1, mat, segments=64):
    profile = ((inner, z0), (outer, z0), (outer, z1-.0004),
               (outer-.0004, z1), (inner+.0002, z1), (inner, z1-.0002))
    vertices = [(x+r*math.cos(2*math.pi*i/segments),
                 y+r*math.sin(2*math.pi*i/segments), z)
                for r, z in profile for i in range(segments)]
    faces = []
    for j in range(len(profile)):
        next_j = (j+1) % len(profile)
        for i in range(segments):
            next_i = (i+1) % segments
            faces.append((j*segments+i, j*segments+next_i,
                          next_j*segments+next_i, next_j*segments+i))
    obj = mesh_object(name, vertices, faces, mat)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def cylinder(name, x, y, radius, z0, z1, mat, sides=64):
    outline = [(x+radius*math.cos(i*2*math.pi/sides),
                y+radius*math.sin(i*2*math.pi/sides)) for i in range(sides)]
    return prism(name, outline, z0, z1, mat)


def drill(obj, x, y, radius, z0, z1):
    cutter = cylinder("BoreTool", x, y, radius, z0, z1, steel)
    bpy.context.view_layer.objects.active = obj
    modifier = obj.modifiers.new("Bore", "BOOLEAN")
    modifier.operation = "DIFFERENCE"
    modifier.solver = "EXACT"
    modifier.object = cutter
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


octagon = [(-.018, -.040), (.018, -.040), (.040, -.018), (.040, .018),
           (.018, .040), (-.018, .040), (-.040, .018), (-.040, -.018)]
housing = prism("OctagonalHousing", octagon, 0, H-.002, dark_steel)
face = prism("FrontPlate", octagon, H-.002, H-.001, steel)
for x in (-PORT_X, PORT_X):
    drill(housing, x, 0, .009, H-.011, H+.001)
    drill(face, x, 0, .009, H-.003, H+.001)

ear_holes = []
for direction, name in ((0, "Top"), (1, "Left"), (2, "Bottom"), (3, "Right")):
    angle = direction*math.pi/2
    c, s = math.cos(angle), math.sin(angle)
    transform = lambda x, y: (c*x-s*y, s*x+c*y)
    outline = [transform(x, y) for x, y in
               ((-.009, .036), (.009, .036), (.009, W/2), (-.009, W/2))]
    ear = prism(f"MountEar{name}", outline, H-.004, H, steel)
    for x in (-.0053, .0053):
        hx, hy = transform(x, .0436)
        drill(ear, hx, hy, .0017, H-.005, H+.001)
        ear_holes.append([hx, hy])

for side, x in (("A", -PORT_X), ("B", PORT_X)):
    annulus(f"Valve{side}OuterTrim", x, 0, .0074, .0106, H-.005, H, polished)
    annulus(f"Valve{side}InnerSeat", x, 0, .0057, .00725, H-.007, H-.0024, steel)
    cylinder(f"Valve{side}DarkSeat", x, 0, .0062, H-.0073, H-.0067, black)
    cylinder(f"Valve{side}Poppet", x, 0, .0033, H-.0067, H-.0057, polished)
    annulus(f"Rear{side}Tube", x, 0, .0021, .0032, -REAR, 0, polished)
    annulus(f"Rear{side}Collar", x, 0, .00325, .0065, -.013462, 0, steel)

for index, (x, y) in enumerate(((-.028, .020), (.028, .020), (-.028, -.020),
                                (.028, -.020), (0, .027), (0, -.027))):
    cylinder(f"Fastener{index}Washer", x, y, .0024, H-.001, H-.0006, polished)
    head = cylinder(f"Fastener{index}Head", x, y, .0019, H-.0006, H, dark_steel, sides=6)
    slot = prism("SlotTool", [(x-.0013, y-.00022), (x+.0013, y-.00022),
                               (x+.0013, y+.00022), (x-.0013, y+.00022)],
                 H-.00025, H+.001, black)
    bpy.context.view_layer.objects.active = head
    modifier = head.modifiers.new("ScrewSlot", "BOOLEAN")
    modifier.operation = "DIFFERENCE"
    modifier.object = slot
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.data.objects.remove(slot, do_unlink=True)

for index, (x, y) in enumerate(((-.026, -.026), (-.026, .026), (.026, -.026), (.026, .026))):
    cylinder(f"RearStud{index}", x, y, .0018, -.009, 0, steel)


def marking(name, text, x, y, size):
    curve = bpy.data.curves.new(name, "FONT")
    curve.body = text
    curve.size = size
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    curve.extrude = .00001
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = visuals
    obj.location = (x, y, H-.0008)
    curve.materials.append(letter)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.convert(target="MESH")


marking("LabelA", "A", -.029, 0, .005)
marking("LabelB", "B", .029, 0, .005)
marking("ModelLabel", "RAFTI", 0, .019, .0035)

reference = {
    "model": "RAFTI Rev A visual research reconstruction",
    "units": "meters", "up_axis": "Z", "outward_axis": "+Z",
    "drawing_dimensions_m": {"front_square_envelope": W, "body_height": H,
                              "rear_tube_dimension": REAR},
    "reconstruction_dimensions_m": {"main_octagon_width": .080, "corner_flat_coordinate": .018,
                                      "valve_center_distance": 2*PORT_X,
                                      "valve_trim_outer_diameter": .0212,
                                      "valve_trim_inner_diameter": .0148,
                                      "ear_width": .018, "ear_thickness": .004,
                                      "ear_bore_diameter": .0034,
                                      "tube_outer_diameter": .0064, "tube_inner_diameter": .0042},
    "mount_hole_xy_m": ear_holes,
    "frames_m": {"Mount": [0, 0, 0], "Mate": [0, 0, H],
                 "FluidA": [-PORT_X, 0, H], "FluidB": [PORT_X, 0, H]},
    "frame_scope": "research rear-body installation plane and front reference plane; manufacturer mating datums not supplied",
    "valve_state": "visual closed valve seats; no fluid-transfer behavior",
    "role": "passive service valve; matching active connector not included",
    "collision_authored": False, "standalone_rigid_body_authored": False,
    "source_pdf": "source/rafti_DOC-00133A.pdf",
    "source_image": "source/dimension_reference.png",
    "mesh_objects": len([o for o in bpy.context.scene.objects if o.type == "MESH"]),
}
(output / "reference_parameters.json").write_text(json.dumps(reference, indent=2) + "\n")
bpy.ops.wm.save_as_mainfile(filepath=str(output / "rafti_reference.blend"))
bpy.ops.wm.usd_export(filepath=str(output / "rafti_reference.usdc"), root_prim_path="/Rafti",
                      export_animation=False, export_materials=True, export_textures=True,
                      generate_preview_surface=True, relative_paths=True)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 32
scene.cycles.use_denoising = True
scene.render.resolution_x = 1000
scene.render.resolution_y = 900
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
world = bpy.data.worlds.new("PreviewWorld")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (.18, .21, .25, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = .65
for index, position in enumerate(((.15, -.18, .23), (-.17, -.05, .10), (.03, .17, -.08))):
    data = bpy.data.lights.new(f"PreviewArea{index}", "AREA")
    data.energy = 8
    data.size = .15
    obj = bpy.data.objects.new(data.name, data)
    scene.collection.objects.link(obj)
    obj.location = position
    obj.rotation_euler = (Vector((0, 0, .02))-obj.location).to_track_quat("-Z", "Y").to_euler()
camera_data = bpy.data.cameras.new("PreviewCamera")
camera_data.type = "ORTHO"
camera = bpy.data.objects.new("PreviewCamera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
for name, position, scale in (("front", (0, 0, .30), .12),
                              ("oblique", (.16, -.20, .17), .16),
                              ("side", (0, -.30, 0), .15),
                              ("rear", (0, 0, -.30), .12)):
    camera.location = position
    camera_data.ortho_scale = scale
    camera.rotation_euler = (Vector((0, 0, 0))-camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(output / f"preview_{name}.png")
    bpy.ops.render.render(write_still=True)
print("RAFTI_REFERENCE_CREATED", json.dumps(reference["drawing_dimensions_m"]))
