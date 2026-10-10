from __future__ import annotations

from typing import Any, Iterable


def metallic_film_parameters() -> dict[str, Any]:
    """Return an adjustable warm aluminized-film rendering approximation.

    These values affect only the USD visual material. They are deliberately
    kept separate from rigid-body mass, inertia, collision and actuator data.
    They are renderer parameters, not measured thermal-optical properties.
    """

    return {
        "base_color_rgb": (0.62, 0.40, 0.12),
        "metallic": 0.72,
        "roughness": 0.34,
        "specular_color_rgb": (0.86, 0.64, 0.32),
        "use_specular_workflow": False,
    }


def solar_panel_parameters() -> dict[str, Any]:
    """Return the visual parameters for the blue photovoltaic panel faces."""

    return {
        "base_color_rgb": (0.018, 0.045, 0.12),
        "metallic": 0.55,
        "roughness": 0.28,
        "specular_color_rgb": (0.70, 0.82, 1.0),
        "use_specular_workflow": False,
    }


def _define_preview_material(
    stage: Any,
    material_path: str,
    params: dict[str, Any],
    *,
    normal_connections: list[Any] | None = None,
) -> Any:
    """Create a PreviewSurface material without changing any physics schema."""

    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*params["base_color_rgb"])
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(params["metallic"])
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(params["roughness"])
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*params["specular_color_rgb"])
    )
    # Explicitly select the metallic workflow. In this workflow
    # specularColor is retained for inspection but does not replace the
    # metallic/base-color controls.
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(
        1 if params.get("use_specular_workflow", False) else 0
    )
    normal_input = shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f)
    if normal_connections:
        normal_input.GetAttr().SetConnections(normal_connections)
    else:
        normal_input.Set(Gf.Vec3f(0.0, 0.0, 1.0))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _normal_connections_for_mesh(mesh: Any) -> list[Any]:
    """Return the original normal-map connections before rebinding a mesh."""

    from pxr import Usd, UsdShade

    original, _ = UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial()
    if not original:
        return []
    for prim in Usd.PrimRange(original.GetPrim()):
        if not prim.IsA(UsdShade.Shader):
            continue
        normal = UsdShade.Shader(prim).GetInput("normal")
        if normal and normal.GetAttr().GetConnections():
            return list(normal.GetAttr().GetConnections())
    return []


def _solar_panel_face_indices(mesh: Any) -> list[int]:
    """Find the two extended photovoltaic faces in a generated Cubesat mesh.

    SimForge's current Cubesat export merges all geometry into one mesh. The
    panels are the high-Z faces extending beyond 45 percent of the mesh's
    Y half-width. The relative thresholds work for both actor and target
    scale variants and fail closed when the mesh has no usable geometry.
    """

    import numpy as np

    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    if points.size == 0 or counts.size == 0 or indices.size == 0:
        return []
    offsets = np.concatenate(([0], np.cumsum(counts)))
    y_limit = 0.45 * float(np.max(np.abs(points[:, 1])))
    z_min = float(np.min(points[:, 2]))
    z_span = float(np.max(points[:, 2]) - z_min)
    if y_limit <= 0.0 or z_span <= 0.0:
        return []
    z_limit = z_min + 0.65 * z_span
    selected: list[int] = []
    for face_index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        face_points = points[indices[start:end]]
        center = face_points.mean(axis=0)
        if abs(float(center[1])) > y_limit and float(center[2]) > z_limit:
            selected.append(face_index)
    return selected


def apply_satellite_materials(
    stage: Any,
    root_prim_paths: Iterable[str],
    *,
    material_path: str = "/World/Materials/srb_satellite_materials",
) -> dict[str, Any]:
    """Bind separate body and photovoltaic materials to current Cubesat meshes.

    A generated Cubesat is a single mesh, so the photovoltaic region is
    authored as a USD face subset. Original normal-map connections are copied
    into replacement shaders where available;
    roughness and base color become explicit nonzero constants. Only USD
    material bindings and subsets are authored. Rigid-body, collision, mass
    and actuator data are untouched.
    """

    from pxr import Usd, UsdGeom, UsdShade

    body_params = metallic_film_parameters()
    panel_params = solar_panel_parameters()
    body_material_paths: list[str] = []
    panel_material_paths: list[str] = []
    bound_meshes: list[str] = []
    panel_faces: dict[str, list[int]] = {}
    normal_inputs_checked = 0
    normal_connections_preserved = 0
    panel_subset_count = 0

    for root_index, root_path in enumerate(root_prim_paths):
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise ValueError(f"invalid material root prim: {root_path}")
        root_suffix = root_path.strip("/").replace("/", "_") or str(root_index)
        body_path = f"{material_path}/body_{root_suffix}"
        panel_path = f"{material_path}/solar_panel_{root_suffix}"
        meshes = [
            UsdGeom.Mesh(prim)
            for prim in Usd.PrimRange(root)
            if prim.IsA(UsdGeom.Mesh)
        ]
        if not meshes:
            continue
        normal_connections = _normal_connections_for_mesh(meshes[0])
        body_material = _define_preview_material(
            stage, body_path, body_params, normal_connections=normal_connections
        )
        panel_material = _define_preview_material(
            stage, panel_path, panel_params, normal_connections=normal_connections
        )
        body_material_paths.append(str(body_material.GetPath()))
        panel_material_paths.append(str(panel_material.GetPath()))
        if normal_connections:
            normal_connections_preserved += 2
        for mesh in meshes:
            UsdShade.MaterialBindingAPI(mesh).Bind(body_material)
            bound_meshes.append(str(mesh.GetPath()))
            normal_inputs_checked += 1
            selected = _solar_panel_face_indices(mesh)
            panel_faces[str(mesh.GetPath())] = selected
            if not selected:
                continue
            subset_path = f"{mesh.GetPath()}/solar_panel_faces"
            existing = stage.GetPrimAtPath(subset_path)
            if existing and existing.IsValid():
                subset = UsdGeom.Subset(existing)
                subset.CreateIndicesAttr(selected)
            else:
                subset = UsdGeom.Subset.CreateGeomSubset(
                    mesh, "solar_panel_faces", UsdGeom.Tokens.face, selected, "materialBind"
                )
            UsdShade.MaterialBindingAPI(subset.GetPrim()).Bind(panel_material)
            panel_subset_count += 1

    return {
        "material_path": material_path,
        "body_material_paths": body_material_paths,
        "panel_material_paths": panel_material_paths,
        "body_material_path": body_material_paths[0] if body_material_paths else None,
        "panel_material_path": panel_material_paths[0] if panel_material_paths else None,
        "body_parameters": body_params,
        "panel_parameters": panel_params,
        "bound_mesh_count": len(bound_meshes),
        "bound_meshes": bound_meshes,
        "panel_faces": panel_faces,
        "panel_face_count": sum(len(indices) for indices in panel_faces.values()),
        "panel_subset_count": panel_subset_count,
        "normal_inputs_checked": normal_inputs_checked,
        "normal_connections_preserved": normal_connections_preserved,
        "roughness_checked": {
            "body": float(body_params["roughness"]),
            "solar_panel": float(panel_params["roughness"]),
        },
        "physics_unchanged_scope": "USD material bindings and face subsets only",
    }


def apply_metallic_film_material(
    stage: Any,
    root_prim_paths: Iterable[str],
    *,
    material_path: str = "/World/Materials/srb_metallic_film",
) -> dict[str, Any]:
    """Bind a gold-like metallic visual material to mesh prims below roots.

    The function uses USD material binding only. It does not touch physics
    schemas or rigid-body properties. Isaac/pxr imports stay lazy so the
    configuration can be unit-tested outside Kit.
    """

    from pxr import Gf, Sdf, Usd, UsdShade

    params = metallic_film_parameters()
    material = UsdShade.Material.Define(stage, material_path)
    shader_path = f"{material_path}/PreviewSurface"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*params["base_color_rgb"])
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(params["metallic"])
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(params["roughness"])
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*params["specular_color_rgb"])
    )
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(
        1 if params.get("use_specular_workflow", False) else 0
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    from pxr import UsdGeom

    bound_meshes: list[str] = []
    for root_path in root_prim_paths:
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise ValueError(f"invalid material root prim: {root_path}")
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh):
                UsdShade.MaterialBindingAPI(prim).Bind(material)
                bound_meshes.append(str(prim.GetPath()))
    return {
        "material_path": material_path,
        "parameters": params,
        "bound_mesh_count": len(bound_meshes),
        "bound_meshes": bound_meshes,
    }
