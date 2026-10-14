"""Verify candidate package hashes, USD contents and handoff availability."""
import hashlib
import json
import sys
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

bundle = Path(sys.argv[1]).resolve()
manifest = json.loads((bundle / "candidate_manifest.json").read_text())
for record in manifest["files"]:
    path = bundle / record["path"]
    assert path.stat().st_size == record["bytes"], path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], path
assert manifest["files"][0]["sha256"] == "fef03075b3b87da9c35e4022a5c39b630404b571d1114b97ca34d7159a9424f1"
stage = Usd.Stage.Open(str(bundle / manifest["visual_usd"]))
assert stage is not None
meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
materials = [p for p in stage.Traverse() if p.IsA(UsdShade.Material)]
triangles = sum(sum(n - 2 for n in UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get()) for p in meshes)
rigid_bodies = sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse())
colliders = sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse())
assets = []
for prim in stage.Traverse():
    for attr in prim.GetAttributes():
        value = attr.Get()
        if isinstance(value, Sdf.AssetPath) and value.path:
            path = Path(value.resolvedPath) if value.resolvedPath else bundle / "visual" / value.path
            assert path.is_file(), path
            assets.append(value.path)
assert str(stage.GetDefaultPrim().GetPath()) == "/Satellite"
assert UsdGeom.GetStageUpAxis(stage) == "Z"
assert len(meshes) == 2 and triangles == 67580 and len(materials) == 32
assert rigid_bodies == 0 and colliders == 0
proposal = json.loads((bundle / "bottom_closure_proposal.json").read_text())
assert proposal["status"] == "proposal_only_not_authored_in_USD"
assert (bundle / "HANDOFF.md").is_file()
for filename in ("whole_oblique.png", "bottom_close_original.png", "bottom_oblique_original.png",
                 "bottom_closure_proposal.png", "rafti_dimensions.png"):
    assert (bundle / "previews" / filename).is_file(), filename
result = {"hash_checks_passed": True, "default_prim": "/Satellite", "up_axis": "Z",
          "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage), "mesh_count": len(meshes),
          "triangles": triangles, "materials": len(materials), "rigid_bodies": rigid_bodies,
          "colliders": colliders, "external_asset_references": assets,
          "physical_scale_calibrated": False, "bottom_cover_authored": False,
          "handoff_and_previews_present": True}
(bundle / "measurements/library_audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
