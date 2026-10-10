"""Local, opt-in orbital illumination and relative photometric diagnostics.

These renderer-scale lights model a plausible visible-light condition. They
do not model spectral irradiance, sensor electrons, or any physical force.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .replay_capture import numpy, write_json


@dataclass(frozen=True)
class OrbitLightingConfig:
    sun_intensity: float = 50000.0
    sun_direction_to_source: tuple = (-1.0, -0.6, 0.8)
    sun_angle_deg: float = 0.53
    sun_temperature_K: float = 5778.0
    earth_intensity: float = 500.0
    earth_direction_to_source: tuple = (0.0, 1.0, -0.5)
    earth_distance_m: float = 100.0
    earth_angular_diameter_deg: float = 136.0
    exposure_time_s: float = 0.004
    film_iso: float = 100.0
    f_number: float = 8.0


SETTINGS = {
    "/rtx/post/histogram/enabled": False,
    "/rtx/post/tonemap/op": 6,  # Installed 4.5 widget: ACES
    "/rtx/post/tonemap/filmIso": 100.0,
    "/rtx/post/tonemap/fNumber": 8.0,
    "/rtx/post/tonemap/whitepoint": [1.0, 1.0, 1.0],
    "/rtx/post/tonemap/colorMode": 0,
    "/rtx/post/tonemap/enableSrgbToGamma": True,
    "/rtx/scenedb/skipMostLights": False,
    "/rtx/sceneDb/ambientLightIntensity": 0.0,
    # Avoid the installed energy-losing firefly clamp in the tested range.
    "/rtx/directLighting/sampledLighting/maxRayIntensity": 1000000.0,
    "/rtx/reflections/sampledLighting/maxRayIntensity": 1000000.0,
    "/rtx/post/dof/enabled": False,
    "/rtx/post/motionblur/enabled": False,
    "/rtx/post/lensFlares/enabled": False,
    "/rtx-transient/dlssg/enabled": False,
}


def captured_world_transform(position, quaternion):
    from pxr import Gf
    matrix = Gf.Matrix4d(1)
    matrix.SetRotate(Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*(float(v) for v in quaternion[1:]))))
    matrix.SetTranslateOnly(Gf.Vec3d(*(float(v) for v in position)))
    return matrix


def physics_snapshot(env):
    from pxr import Usd
    bodies = {}
    for name in ("_robot", "_target"):
        view = getattr(env, name).root_physx_view
        bodies[name] = {key: numpy(getattr(view, f"get_{key}")()).tolist()
                        for key in ("masses", "inertias", "coms", "transforms", "velocities")}
    attrs = {}
    for prim in env.sim.stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetName().startswith(("physics:", "physx")):
                attrs[str(attr.GetPath())] = str(attr.Get())
    term = env.action_manager._terms[env._thrust_action_term_key]
    return {"bodies": bodies, "physics_attributes": attrs,
            "fuel": numpy(term.remaining_fuel).tolist(),
            "actuator_cfg": str(term.cfg),
            "thruster_power": numpy(term._thruster_power).tolist(),
            "thruster_direction": numpy(term._thruster_direction).tolist(),
            "thruster_offset": numpy(term._thruster_offset).tolist(),
            "sim_time_s": float(env.sim.current_time)}


class OrbitLightingTrial:
    def __init__(self, env, capture, config):
        import carb.settings
        from pxr import UsdLux

        self.env, self.capture, self.config = env, capture, config
        self.output = capture.output
        self.settings = carb.settings.get_settings()
        self.original_lights = []
        for prim in env.sim.stage.Traverse():
            if prim.HasAPI(UsdLux.LightAPI):
                light = UsdLux.LightAPI(prim)
                self.original_lights.append((light, light.GetIntensityAttr().Get()))
        self.initial_settings = self.runtime_settings()
        for key, value in SETTINGS.items():
            self.settings.set(key, value)
        self.settings.set("/rtx/post/tonemap/exposureTime", config.exposure_time_s)
        self.settings.set('/rtx/post/tonemap/fNumber', config.f_number)
        self.settings.set('/rtx/post/tonemap/filmIso', config.film_iso)
        self.before = physics_snapshot(env)
        self.calibration = []

    def runtime_settings(self):
        keys = [*SETTINGS, "/rtx/post/tonemap/exposureTime", "/rtx/post/tonemap/cm2Factor",
                "/rtx/rendermode", "/rtx/reflections/enabled", "/rtx/indirectDiffuse/enabled",
                "/rtx/shadows/enabled", "/rtx/ambientOcclusion/enabled",
                "/rtx/raytracing/fractionalCutoutOpacity", "/rtx/pathtracing/spp",
                "/physics/fabricUpdateTransformations", "/physics/updateToUsd",
                "/rtx/directLighting/sampledLighting/maxRayIntensity", "/rtx/reflections/sampledLighting/maxRayIntensity",
                "/app/useFabricSceneDelegate", "/persistent/omnihydra/useFastSceneDelegate"]
        keys += ['/rtx/sceneDb/ambientLightColor', '/rtx/sceneDb/ambientLightIntensity', '/rtx/post/aa/op']
        return {key: self.settings.get(key) for key in keys}

    def install(self):
        from pxr import Gf, UsdGeom, UsdLux
        for light, _ in self.original_lights:
            light.GetIntensityAttr().Set(0.0)
        stage, cfg = self.env.sim.stage, self.config
        self.sun = UsdLux.DistantLight.Define(stage, "/World/orbit_trial/sun")
        self.sun.CreateIntensityAttr(cfg.sun_intensity)
        self.sun.CreateExposureAttr(0.0)
        self.sun.CreateAngleAttr(cfg.sun_angle_deg)
        self.sun.CreateEnableColorTemperatureAttr(True)
        self.sun.CreateColorTemperatureAttr(cfg.sun_temperature_K)
        direction = np.asarray(cfg.sun_direction_to_source, dtype=float)
        direction /= np.linalg.norm(direction)
        q = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*direction)).GetQuat()
        UsdGeom.Xformable(self.sun.GetPrim()).AddOrientOp().Set(Gf.Quatf(q))
        self.earth = UsdLux.DiskLight.Define(stage, "/World/orbit_trial/earth_reflection")
        self.earth.CreateIntensityAttr(0.0)
        self.earth.CreateExposureAttr(0.0)
        self.earth.CreateNormalizeAttr(False)
        self.earth.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        radius = cfg.earth_distance_m*np.tan(np.deg2rad(cfg.earth_angular_diameter_deg/2))
        self.earth.CreateRadiusAttr(float(radius))
        direction_e = np.asarray(cfg.earth_direction_to_source, dtype=float)
        direction_e /= np.linalg.norm(direction_e)
        pos = numpy(self.env._target.data.root_pos_w[0])+cfg.earth_distance_m*direction_e
        xf = UsdGeom.Xformable(self.earth.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        q = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*direction_e)).GetQuat()
        xf.AddOrientOp().Set(Gf.Quatf(q))
        self.radius = radius
        self.directions = {"sun_to_source": direction, "earth_to_source": direction_e}

    def set_profile(self, name):
        if name not in ('original', 'sun_only', 'sun_earth', 'earth_only', 'lights_off'):
            raise ValueError(f'unknown lighting profile: {name}')
        self.settings.set('/rtx/sceneDb/ambientLightIntensity',
                          self.initial_settings.get('/rtx/sceneDb/ambientLightIntensity', 0.) if name == 'original' else 0.)
        for light, value in self.original_lights:
            # Remove stock illumination, including textured Dome reflection
            # paths, from the active scene rather than relying on intensity=0.
            light.GetPrim().SetActive(name == 'original')
            light.GetIntensityAttr().Set(value if name == "original" else 0.0)
        self.sun.GetIntensityAttr().Set(self.config.sun_intensity if name in ('sun_only', 'sun_earth') else 0.0)
        self.earth.GetIntensityAttr().Set(self.config.earth_intensity if name in ('sun_earth', 'earth_only') else 0.0)
        self.active_profile = name

    def capture_profile(self, name):
        """Change lights and capture at the existing state, without stepping physics."""
        from pxr import UsdLux
        before = physics_snapshot(self.env)
        self.set_profile(name)
        self.capture.settle(30)
        self.capture.snapshot(name)
        after = physics_snapshot(self.env)
        lights = []
        for prim in self.env.sim.stage.Traverse():
            if prim.HasAPI(UsdLux.LightAPI):
                light = UsdLux.LightAPI(prim)
                lights.append({'path': str(prim.GetPath()), 'type': prim.GetTypeName(),
                               'intensity': light.GetIntensityAttr().Get(),
                               'exposure': light.GetExposureAttr().Get()})
        write_json(self.output/'static'/f'{name}_lighting.json', {
            'profile': name, 'lights': lights, 'directions_to_source': self.directions,
            'stock_lights': [{'path': str(light.GetPrim().GetPath()),
                             'active': light.GetPrim().IsActive(), 'intensity': light.GetIntensityAttr().Get()}
                            for light, _ in self.original_lights],
            'runtime_settings': self.runtime_settings(),
            'physics_before': before, 'physics_after': after, 'physics_unchanged': before == after,
            'earth_only_scope': 'direct Sun disabled; uniform Earth disk surrogate, not a validated orbital umbra/phase geometry',
        })
        self.export_scene(self.output/'static'/f'{name}.usda')

    def add_grayboard(self):
        from pxr import Gf, Sdf, UsdGeom, UsdShade
        from srb.nmpc.quaternion import quaternion_to_rotation
        data = self.capture.camera.data
        pos = numpy(data.pos_w[0])
        rot = quaternion_to_rotation(numpy(data.quat_w_world[0]))
        # Plane normal toward camera (-X), centered one meter in front.
        points = np.array([[1., -.18, -.18], [1., -.18, .18],
                           [1., .18, .18], [1., .18, -.18]])@rot.T+pos
        mesh = UsdGeom.Mesh.Define(self.env.sim.stage, "/World/orbit_trial/grayboard")
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateDoubleSidedAttr(True)
        material = UsdShade.Material.Define(self.env.sim.stage, "/World/orbit_trial/gray_material")
        shader = UsdShade.Shader.Define(self.env.sim.stage, "/World/orbit_trial/gray_material/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.18, .18, .18))
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    def measure_gray(self, name):
        import imageio.v2 as imageio
        self.capture.settle(20)
        rgb, _, _ = self.capture.images()
        h, w = rgb.shape[:2]
        roi = rgb[h//2-12:h//2+12, w//2-12:w//2+12, :3]
        imageio.imwrite(self.output/"static"/f"gray_{name}.png", rgb)
        result = {"name": name, "sun_intensity": self.sun.GetIntensityAttr().Get(),
                  "earth_intensity": self.earth.GetIntensityAttr().Get(),
                  "exposure_time_s": self.settings.get("/rtx/post/tonemap/exposureTime"),
                  "tonemap_op": self.settings.get("/rtx/post/tonemap/op"),
                  "roi_xyxy": [w//2-12, h//2-12, w//2+11, h//2+11],
                  "ldr_mean_rgb": roi.mean((0, 1)), "ldr_saturated_fraction": float(np.mean(roi >= 250))}
        if self.capture.hdr is not None:
            try:
                hdr = np.asarray(self.capture.hdr.get_data()).copy()
                if hdr.shape[:2] != rgb.shape[:2] or not np.issubdtype(hdr.dtype, np.floating):
                    raise ValueError(f"unexpected HDR buffer {hdr.shape}, {hdr.dtype}")
                np.save(self.output/"static"/f"gray_{name}_hdr.npy", hdr)
                linear = hdr[h//2-12:h//2+12, w//2-12:w//2+12, :3].astype(float)
                result.update(hdr_dtype=str(hdr.dtype), hdr_mean_rgb=linear.mean((0, 1)),
                              hdr_mean_luma=float(np.mean(linear@np.array([.2126, .7152, .0722]))))
                for aov in ('direct_diffuse', 'diffuse_albedo'):
                    values = np.asarray(getattr(self.capture, aov).get_data()).copy()
                    np.save(self.output/'static'/f'gray_{name}_{aov}.npy', values)
                    ah, aw = values.shape[:2]
                    dy, dx = max(1, round(12*ah/h)), max(1, round(12*aw/w))
                    patch = values[ah//2-dy:ah//2+dy, aw//2-dx:aw//2+dx, :3].astype(float)
                    result[aov+'_mean_rgb'] = patch.mean((0, 1))
                    result[aov+'_dtype'] = str(values.dtype)
                    result[aov+'_shape'] = list(values.shape)
            except Exception as exc:
                result["hdr_error"] = repr(exc)
        else:
            result["hdr_error"] = self.capture.hdr_error
        self.calibration.append(result)
        write_json(self.output/"grayboard_measurements.json", self.calibration)

    def calibrate(self):
        self.add_grayboard()
        self.earth.GetIntensityAttr().Set(0.0)
        # Diagnose the AOV pipeline with the installed Clamp operator, which
        # skips exposure/tone mapping. Float storage alone does not prove HDR.
        self.settings.set("/rtx/post/tonemap/op", 0)
        for factor in (.001, .01, .1, 1., 2.):
            self.sun.GetIntensityAttr().Set(self.config.sun_intensity*factor)
            self.measure_gray(f"clamp_sun_{factor:g}x")
        self.settings.set("/rtx/post/tonemap/op", SETTINGS["/rtx/post/tonemap/op"])
        for factor in (1., 2., 4.):
            self.sun.GetIntensityAttr().Set(self.config.sun_intensity*factor)
            self.measure_gray(f"sun_{factor:g}x")
        self.sun.GetIntensityAttr().Set(self.config.sun_intensity)
        for factor in (.5, 1., 2.):
            self.settings.set("/rtx/post/tonemap/exposureTime", self.config.exposure_time_s*factor)
            self.measure_gray(f"exposure_{factor:g}x")
        self.settings.set("/rtx/post/tonemap/exposureTime", self.config.exposure_time_s)
        self.sun.GetIntensityAttr().Set(0.0)
        for factor in (0., 1., 10.):
            self.earth.GetIntensityAttr().Set(self.config.earth_intensity*factor)
            self.measure_gray(f"earth_only_{factor:g}x")
        self.env.sim.stage.RemovePrim("/World/orbit_trial/grayboard")
        self.env.sim.stage.RemovePrim("/World/orbit_trial/gray_material")
        self.set_profile("sun_earth")

    def compare_fov(self):
        original = self.capture.camera.data.intrinsic_matrices.clone()
        focal = self.capture.camera._sensor_prims[0].GetFocalLengthAttr().Get()
        for fov in (45., 60.):
            intrinsic = original.clone()
            intrinsic[:, 0, 0] = self.capture.camera.cfg.width/(2*np.tan(np.deg2rad(fov/2)))
            intrinsic[:, 1, 1] = intrinsic[:, 0, 0]
            self.capture.camera.set_intrinsic_matrices(intrinsic, focal_length=focal)
            self.capture.settle(20)
            self.capture.snapshot(f'fov_{fov:g}')
        self.capture.camera.set_intrinsic_matrices(original, focal_length=focal)

    def save(self):
        from pxr import UsdGeom, UsdLux, UsdShade
        after = physics_snapshot(self.env)
        write_json(self.output/"visual_physics_invariance.json", {
            "before": self.before, "after": after, "unchanged": self.before == after,
            "scope": "static lighting/material/semantic capture setup; fuel burn during later control is expected",
        })
        config = asdict(self.config)
        config.update(directions=self.directions, earth_radius_m=float(self.radius),
                      active_profile=getattr(self, 'active_profile', None),
                      effective_sun_intensity=self.sun.GetIntensityAttr().Get(),
                      effective_earth_intensity=self.earth.GetIntensityAttr().Get(),
                      light_intensity_units="uncalibrated renderer scale; not W/m^2",
                      earth_model="uniform distant disk surrogate; no phase map, atmosphere, orbit propagation or thermal IR",
                      earth_angular_size_assumption="136 deg illustrative LEO geometry, approximately 500 km; task orbit is unspecified",
                      sun_model="single DistantLight, RGB 5778 K approximation, finite 0.53 deg solar disk",
                      original_lights=[{"path": str(light.GetPrim().GetPath()), "intensity_before": value,
                                       "active_after": light.GetPrim().IsActive(),
                                       "intensity_after": light.GetIntensityAttr().Get()}
                                      for light, value in self.original_lights],
                      initial_runtime_settings=self.initial_settings, effective_runtime_settings=self.runtime_settings(),
                      physics_unchanged=self.before == after)
        write_json(self.output/"orbit_lighting_config.json", config)
        inventory = {"meters_per_unit": UsdGeom.GetStageMetersPerUnit(self.env.sim.stage),
                     "up_axis": str(UsdGeom.GetStageUpAxis(self.env.sim.stage)),
                     "lights": [], "meshes": [], "cameras": []}
        for prim in self.env.sim.stage.Traverse():
            if prim.HasAPI(UsdLux.LightAPI):
                inventory["lights"].append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                    "attrs": {a.GetName(): str(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith("inputs:")}})
            elif prim.IsA(UsdGeom.Mesh):
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                inventory["meshes"].append({"path": str(prim.GetPath()), "material": str(material.GetPath()) if material else None})
            elif prim.IsA(UsdGeom.Camera):
                inventory["cameras"].append({"path": str(prim.GetPath()),
                    "attrs": {a.GetName(): str(a.Get()) for a in prim.GetAttributes()}})
        write_json(self.output/"scene_inventory.json", inventory)
        self.export_scene(self.output/"scene_initial.usda")

    def export_scene(self, path):
        # Export an independent flattened layer; bake actual physical body poses
        # into the copy because Fabric transforms are not USD authored opinions.
        from pxr import Usd, UsdGeom, UsdPhysics
        stage_copy = Usd.Stage.Open(self.env.sim.stage.Flatten())
        cache = UsdGeom.XformCache()
        for asset, root in ((self.env._robot, "robot"), (self.env._target, "target")):
            body = next(p for p in Usd.PrimRange(stage_copy.GetPrimAtPath(f"/World/envs/env_0/{root}"))
                        if p.HasAPI(UsdPhysics.RigidBodyAPI))
            q, pos = numpy(asset.data.root_quat_w[0]), numpy(asset.data.root_pos_w[0])
            world = captured_world_transform(pos, q)
            local = world*cache.GetLocalToWorldTransform(body.GetParent()).GetInverse()
            xf = UsdGeom.Xformable(body)
            xf.ClearXformOpOrder()
            xf.AddTransformOp(opSuffix="capturedPose").Set(local)
        stage_copy.GetRootLayer().Export(str(path))
