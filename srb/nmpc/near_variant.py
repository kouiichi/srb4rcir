"""Build the derived NEAR model with its central antenna replaced by a boss."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


ANTENNA_MIN_Z = 1.12
ANTENNA_MIN_BASE_Z = 0.50
ANTENNA_MAX_CENTER_RADIUS = 0.35


def _mesh_components(stage, root_prim, prim):
    """Return root-space vertex components and their corresponding face ids."""
    from pxr import Gf, UsdGeom

    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    transform, _ = UsdGeom.XformCache().ComputeRelativeTransform(prim, root_prim)
    root_points = np.asarray([
        transform.Transform(Gf.Vec3d(*map(float, point))) for point in points
    ], dtype=float)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    faces = []
    edges = []
    offset = 0
    for face_id, count in enumerate(counts):
        face = indices[offset:offset + count]
        offset += count
        faces.append(face)
        if len(face) > 1:
            edges.extend(zip(face, np.roll(face, 1)))
    if not edges:
        return root_points, faces, np.arange(len(points), dtype=int), []
    edge_array = np.asarray(edges, dtype=int)
    graph = coo_matrix((np.ones(len(edge_array)),
                        (edge_array[:, 0], edge_array[:, 1])),
                       shape=(len(points), len(points)))
    _, labels = connected_components(graph, directed=False)
    records = []
    for label in np.unique(labels):
        vertex_ids = np.flatnonzero(labels == label)
        component_points = root_points[vertex_ids]
        face_ids = [i for i, face in enumerate(faces)
                    if len(face) and labels[face[0]] == label]
        if not face_ids:
            continue
        records.append({
            'label': int(label),
            'vertex_ids': vertex_ids,
            'face_ids': np.asarray(face_ids, dtype=int),
            'bounds_min': component_points.min(axis=0),
            'bounds_max': component_points.max(axis=0),
            'center_xy': (component_points[:, :2].min(axis=0)
                          + component_points[:, :2].max(axis=0)) / 2,
            'face_count': len(face_ids),
        })
    return root_points, faces, labels, records


def _mesh_inventory(stage, root_prim):
    from pxr import UsdGeom

    inventory = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            points, faces, labels, components = _mesh_components(stage, root_prim, prim)
            inventory.append((prim, points, faces, labels, components))
    return inventory


def _select_bus_component(inventory):
    candidates = []
    for prim, _, _, _, components in inventory:
        for component in components:
            low, high = component['bounds_min'], component['bounds_max']
            dimensions = high - low
            if (np.linalg.norm(component['center_xy']) < 0.15
                    and 1.0 <= dimensions[0] <= 2.0
                    and 1.0 <= dimensions[1] <= 2.0
                    and dimensions[2] > 1.0
                    and high[2] < 1.0):
                candidates.append((dimensions[2], prim, component))
    if not candidates:
        raise ValueError('could not identify the central NEAR bus component')
    _, prim, component = max(candidates, key=lambda item: item[0])
    return prim, component


def _select_deck_component(inventory):
    candidates = []
    for prim, _, _, _, components in inventory:
        if prim.GetPath().name != 'NEAR_023':
            continue
        for component in components:
            low, high = component['bounds_min'], component['bounds_max']
            dimensions = high - low
            if (np.linalg.norm(component['center_xy']) < 0.15
                    and 1.4 <= dimensions[0] <= 1.6
                    and 1.4 <= dimensions[1] <= 1.6
                    and 0.70 <= high[2] <= 0.80):
                candidates.append((dimensions[2], prim, component))
    if not candidates:
        raise ValueError('could not identify the central NEAR top-deck component')
    _, prim, component = min(candidates, key=lambda item: item[0])
    return prim, component


def _write_filtered_mesh(mesh, faces, labels, removed_labels):
    """Author face/component removal while retaining normals and primvars."""
    from pxr import Gf, Vt, UsdGeom

    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    face_keep = np.asarray([
        not (len(face) and labels[face[0]] in removed_labels) for face in faces
    ], dtype=bool)
    corner_keep = np.repeat(face_keep, counts)
    kept_corners = indices[corner_keep]
    used_vertices = np.unique(kept_corners)
    vertex_remap = np.full(len(labels), -1, dtype=int)
    vertex_remap[used_vertices] = np.arange(len(used_vertices))
    new_points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)[used_vertices]
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(*map(float, point)) for point in new_points
    ]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(counts[face_keep].tolist()))
    remapped_corners = vertex_remap[kept_corners]
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(remapped_corners.tolist()))
    mesh.GetExtentAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(*map(float, new_points.min(axis=0))),
        Gf.Vec3f(*map(float, new_points.max(axis=0))),
    ]))

    normals_attr = mesh.GetNormalsAttr()
    normals = normals_attr.Get()
    if normals is not None:
        interpolation = str(mesh.GetNormalsInterpolation())
        if interpolation == 'faceVarying':
            filtered = np.asarray(normals)[corner_keep]
        elif interpolation in ('vertex', 'varying'):
            filtered = np.asarray(normals)[used_vertices]
        else:
            filtered = np.asarray(normals)
        normals_attr.Set(Vt.Vec3fArray([
            Gf.Vec3f(*map(float, normal)) for normal in filtered
        ]))

    for primvar in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
        values = primvar.Get()
        if values is None:
            continue
        interpolation = str(primvar.GetInterpolation())
        indexed = primvar.IsIndexed()
        if indexed:
            old_indices = np.asarray(primvar.GetIndices(), dtype=int)
            if interpolation == 'faceVarying':
                primvar.SetIndices(Vt.IntArray(old_indices[corner_keep].tolist()))
            elif interpolation == 'uniform':
                primvar.SetIndices(Vt.IntArray(old_indices[face_keep].tolist()))
            elif interpolation in ('vertex', 'varying'):
                primvar.SetIndices(Vt.IntArray(old_indices[used_vertices].tolist()))
        elif interpolation == 'faceVarying':
            primvar.Set(type(values)([values[int(i)] for i in np.flatnonzero(corner_keep)]))
        elif interpolation == 'uniform':
            primvar.Set(type(values)([values[int(i)] for i in np.flatnonzero(face_keep)]))
        elif interpolation in ('vertex', 'varying'):
            primvar.Set(type(values)([values[int(i)] for i in used_vertices]))


def _define_boss(stage, center_xy, side, bottom_z, height):
    from pxr import Gf, Sdf, Vt, UsdGeom, UsdShade

    path = '/Satellite/antenna_replacement_boss'
    mesh = UsdGeom.Mesh.Define(stage, path)
    half = float(side) / 2
    x0, x1 = float(center_xy[0]) - half, float(center_xy[0]) + half
    y0, y1 = float(center_xy[1]) - half, float(center_xy[1]) + half
    z0, z1 = float(bottom_z), float(bottom_z + height)
    points = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
              (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
    faces = [0,3,2,1, 4,5,6,7, 0,1,5,4, 1,2,6,5,
             2,3,7,6, 3,0,4,7]
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(*map(float, point)) for point in points
    ]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4] * 6))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.GetPrim().SetCustomDataByKey('targetVariant', 'near_antenna_boss')
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        'st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    uv_face = [Gf.Vec2f(0,0), Gf.Vec2f(1,0), Gf.Vec2f(1,1), Gf.Vec2f(0,1)]
    st.Set(Vt.Vec2fArray(uv_face * 6))
    material = UsdShade.Material.Get(stage, '/Satellite/_materials/foil_gold')
    if material:
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
    return mesh


def build_near_variant(source_usd, output_usd):
    """Create a reproducible USD derivative and its compact geometry manifest."""
    from pxr import Gf, Usd, UsdGeom

    source = Path(source_usd).resolve()
    output = Path(output_usd).resolve()
    if source == output:
        raise ValueError('derived NEAR USD must not overwrite the original asset')
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.tmp' + output.suffix)
    if temporary.exists():
        temporary.unlink()

    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    original = Usd.Stage.Open(str(source))
    original_root = original.GetDefaultPrim()
    if not original_root:
        raise ValueError(f'NEAR source has no default prim: {source}')
    stage = Usd.Stage.CreateNew(str(temporary))
    root = UsdGeom.Xform.Define(stage, str(original_root.GetPath()))
    stage.SetDefaultPrim(root.GetPrim())
    source_reference = os.path.relpath(source, temporary.parent)
    root.GetPrim().GetReferences().AddReference(source_reference)

    inventory = _mesh_inventory(stage, root.GetPrim())
    bus_prim, bus = _select_bus_component(inventory)
    bus_low, bus_high = bus['bounds_min'], bus['bounds_max']
    bus_dimensions = bus_high - bus_low
    deck_prim, deck = _select_deck_component(inventory)
    deck_low, deck_high = deck['bounds_min'], deck['bounds_max']
    deck_dimensions = deck_high - deck_low
    side = float(max(deck_dimensions[0], deck_dimensions[1]))
    center_xy = (deck_low[:2] + deck_high[:2]) / 2
    # Preserve the source solar-array reference used by existing rendezvous tasks.
    panel_path = original_root.GetPath().AppendChild('shiny_panels')
    panel_prim = stage.GetPrimAtPath(panel_path)
    panel_mesh = next((p for p in Usd.PrimRange(panel_prim) if p.IsA(UsdGeom.Mesh)), None)
    if panel_mesh is None:
        raise ValueError('NEAR solar-panel reference has no mesh')
    panel_root = stage.GetPrimAtPath(original_root.GetPath())
    panel_points = np.asarray(UsdGeom.Mesh(panel_mesh).GetPointsAttr().Get(), dtype=float)
    panel_transform, _ = UsdGeom.XformCache().ComputeRelativeTransform(panel_mesh, panel_root)
    panel_points = np.asarray([
        panel_transform.Transform(Gf.Vec3d(*map(float, point))) for point in panel_points
    ])
    if np.ptp(panel_points[:, 2]) > 1e-5:
        raise ValueError('NEAR solar-array reference is not planar')
    deck_z = float(np.mean(panel_points[:, 2]))
    boss_height = float(bus_dimensions[2] / 20)

    removed = []
    for prim, points, faces, labels, components in inventory:
        remove_labels = set()
        for component in components:
            low, high = component['bounds_min'], component['bounds_max']
            center_radius = float(np.linalg.norm(component['center_xy'] - center_xy))
            if (high[2] >= ANTENNA_MIN_Z
                    and low[2] >= ANTENNA_MIN_BASE_Z
                    and center_radius <= ANTENNA_MAX_CENTER_RADIUS):
                remove_labels.add(component['label'])
                removed.append({
                    'mesh_path': str(prim.GetPath()),
                    'component_id': component['label'],
                    'vertex_count': int(np.count_nonzero(labels == component['label'])),
                    'face_count': component['face_count'],
                    'bounds_min_root': low.tolist(),
                    'bounds_max_root': high.tolist(),
                })
        if remove_labels:
            _write_filtered_mesh(UsdGeom.Mesh(prim), faces, labels, remove_labels)

    if len(removed) != 9:
        raise ValueError(f'expected 9 central antenna components; identified {len(removed)}')
    _define_boss(stage, center_xy, side, deck_z, boss_height)
    root.GetPrim().SetCustomDataByKey('targetVariant', 'near_antenna_boss')
    root.GetPrim().SetCustomDataByKey('sourceSha256', source_hash)
    root.GetPrim().SetCustomDataByKey('antennaRemovedComponentCount', len(removed))
    stage.GetRootLayer().Save()
    del stage
    os.replace(temporary, output)

    manifest = {
        'generator_version': 1,
        'target_variant': 'near_antenna_boss',
        'source_usd': str(source),
        'source_sha256': source_hash,
        'derived_usd': str(output),
        'bus_mesh_path': str(bus_prim.GetPath()),
        'deck_mesh_path': str(deck_prim.GetPath()),
        'deck_dimensions_source_units': deck_dimensions.tolist(),
        'bus_dimensions_source_units': bus_dimensions.tolist(),
        'bus_side_source_units': float(max(bus_dimensions[0], bus_dimensions[1])),
        'deck_side_source_units': side,
        'bus_height_source_units': float(bus_dimensions[2]),
        'boss_dimensions_source_units': [side, side, boss_height],
        'boss_height_fraction_of_bus': 1/20,
        'boss_bottom_z_source_units': deck_z,
        'boss_top_z_source_units': deck_z + boss_height,
        'deck_top_z_source_units': deck_z,
        'removed_component_count': len(removed),
        'removed_components': removed,
    }
    manifest_path = output.with_suffix('.metadata.json')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
