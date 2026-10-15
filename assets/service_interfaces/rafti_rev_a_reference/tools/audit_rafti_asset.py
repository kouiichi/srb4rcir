"""Check the generated reference asset's scale, datums and composability."""
import json
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

folder = Path(sys.argv[1]).resolve()
path = folder / "rafti_reference.usdc"
assert path.is_file(), "RAFTI reference USD has not been generated"
stage = Usd.Stage.Open(str(path))
assert stage and str(stage.GetDefaultPrim().GetPath()) == "/Rafti"
assert UsdGeom.GetStageUpAxis(stage) == "Z"
assert UsdGeom.GetStageMetersPerUnit(stage) == 1
cache = UsdGeom.XformCache()
vertices = []
meshes = []
for prim in Usd.PrimRange(stage.GetPrimAtPath("/Rafti/Visuals")):
    if prim.IsA(UsdGeom.Mesh):
        meshes.append(prim)
        transform = cache.GetLocalToWorldTransform(prim)
        vertices.extend([list(transform.Transform(Gf.Vec3d(v)))
                         for v in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
vertices = np.asarray(vertices)
lower, upper = vertices.min(axis=0), vertices.max(axis=0)
np.testing.assert_allclose(upper[:2] - lower[:2], [.096012, .096012], atol=2e-6)
np.testing.assert_allclose([lower[2], upper[2]], [-.0516128, .048260], atol=2e-6)
frames = {}
for name in ("Mount", "Mate", "FluidA", "FluidB"):
    prim = stage.GetPrimAtPath(f"/Rafti/Frames/{name}")
    assert prim.IsValid(), name
    transform = cache.GetLocalToWorldTransform(prim)
    frames[name] = list(transform.ExtractTranslation())
np.testing.assert_allclose(frames["Mount"], [0, 0, 0], atol=1e-7)
np.testing.assert_allclose(frames["Mate"], [0, 0, .048260], atol=1e-7)
assert frames["FluidA"][0] < 0 < frames["FluidB"][0]
assert all(np.isfinite(vertices.ravel()))
assert sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse()) == 0
mounted = Usd.Stage.CreateInMemory()
host = UsdGeom.Xform.Define(mounted, "/Host")
attachment = UsdGeom.Xform.Define(mounted, "/Host/Port")
attachment.AddTranslateOp().Set(Gf.Vec3d(.1, .2, .3))
attachment.GetPrim().GetReferences().AddReference(str(path))
assert mounted.GetPrimAtPath("/Host/Port/Frames/Mount").IsValid()
np.testing.assert_allclose(UsdGeom.XformCache().GetLocalToWorldTransform(
    mounted.GetPrimAtPath("/Host/Port/Frames/Mount")).ExtractTranslation(), [.1, .2, .3])
result = {"mesh_count": len(meshes), "bounds_m": [lower.tolist(), upper.tolist()],
          "frames_m": frames, "reference_composition_passed": True,
          "rigid_body_count": 0, "physical_contact_validated": False}
(folder / "usd_audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
