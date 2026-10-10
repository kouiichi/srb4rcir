from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def extract_rgb_depth(output: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Extract raw RGB and depth from Isaac camera output.

    Isaac Lab versions expose depth either as ``depth`` or as
    ``distance_to_image_plane``. Both forms are accepted.
    """

    if "rgb" not in output:
        raise KeyError("camera output does not contain rgb")
    depth_key = "depth" if "depth" in output else "distance_to_image_plane"
    if depth_key not in output:
        raise KeyError("camera output does not contain depth or distance_to_image_plane")
    return np.asarray(output["rgb"]), np.asarray(output[depth_key])


def project_target_visible(
    target_world: Sequence[float],
    view_projection: np.ndarray,
    width: int,
    height: int,
) -> bool:
    """Check whether a world point projects inside the camera clip rectangle."""

    point = np.ones(4, dtype=np.float64)
    point[:3] = np.asarray(target_world, dtype=np.float64)
    clip = np.asarray(view_projection, dtype=np.float64) @ point
    if clip.shape != (4,) or not np.all(np.isfinite(clip)) or clip[3] <= 0.0:
        return False
    ndc = clip[:3] / clip[3]
    return bool(
        -1.0 <= ndc[0] <= 1.0
        and -1.0 <= ndc[1] <= 1.0
        and -1.0 <= ndc[2] <= 1.0
        and width > 0
        and height > 0
    )


def camera_frame_target_visible(
    target_world: Sequence[float],
    camera_pos_world: Sequence[float],
    camera_quat_wxyz: Sequence[float],
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> bool:
    """Project a point using Isaac Lab's world camera frame (+X forward, +Z up)."""

    q = np.asarray(camera_quat_wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        return False
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    rotation_world_camera = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    relative = rotation_world_camera.T @ (
        np.asarray(target_world, dtype=np.float64)
        - np.asarray(camera_pos_world, dtype=np.float64)
    )
    if relative.shape != (3,) or not np.all(np.isfinite(relative)) or relative[0] <= 0:
        return False
    k = np.asarray(intrinsic, dtype=np.float64)
    if k.shape != (3, 3) or not np.all(np.isfinite(k)):
        return False
    u = k[0, 0] * (-relative[1] / relative[0]) + k[0, 2]
    v = k[1, 1] * (-relative[2] / relative[0]) + k[1, 2]
    return bool(0.0 <= u < width and 0.0 <= v < height)


def summarize_camera_frame(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    clipping_range: Sequence[float],
    target_visible: bool | None = None,
) -> dict[str, Any]:
    """Return deterministic quality metrics for one RGB/depth camera frame.

    ``depth_valid_fraction`` only counts finite samples inside the configured
    camera clipping range.  It deliberately does not infer object identity;
    ``target_visible`` is supplied by the caller's geometric visibility test.
    """

    rgb_array = np.asarray(rgb)
    depth_array = np.asarray(depth)
    if rgb_array.ndim != 3 or rgb_array.shape[-1] < 3:
        raise ValueError(f"rgb must have shape (H, W, >=3), got {rgb_array.shape}")
    if depth_array.ndim == 3 and depth_array.shape[-1] == 1:
        depth_array = depth_array[..., 0]
    if depth_array.ndim != 2 or depth_array.shape[:2] != rgb_array.shape[:2]:
        raise ValueError(
            "depth must have shape (H, W) matching rgb, "
            f"got rgb={rgb_array.shape}, depth={depth_array.shape}"
        )

    rgb_float = rgb_array[..., :3].astype(np.float64, copy=False)
    finite_rgb = np.isfinite(rgb_float)
    low, high = (float(clipping_range[0]), float(clipping_range[1]))
    if not np.isfinite(low) or not np.isfinite(high) or not low < high:
        raise ValueError(f"invalid clipping_range={clipping_range}")

    finite_depth = np.isfinite(depth_array)
    valid_depth = finite_depth & (depth_array >= low) & (depth_array <= high)
    pixel_count = int(rgb_array.shape[0] * rgb_array.shape[1])
    overexposed = np.any(rgb_float >= 250.0, axis=-1)
    luma = (0.2126 * rgb_float[..., 0] +
            0.7152 * rgb_float[..., 1] +
            0.0722 * rgb_float[..., 2])
    underexposed = luma <= 5.0

    result: dict[str, Any] = {
        "rgb_shape": list(rgb_array.shape),
        "depth_shape": list(depth_array.shape),
        "rgb_finite": bool(np.all(finite_rgb)),
        "depth_finite": bool(np.all(finite_depth)),
        "overexposed_fraction": float(np.count_nonzero(overexposed) / pixel_count),
        "underexposed_fraction": float(np.count_nonzero(underexposed) / pixel_count),
        "nonblack_fraction": float(np.count_nonzero(luma > 5.0) / pixel_count),
        "mean_luma_u8": float(np.mean(luma)),
        "max_luma_u8": float(np.max(luma)),
        "depth_valid_fraction": float(np.count_nonzero(valid_depth) / pixel_count),
        "depth_min_m": float(np.min(depth_array[valid_depth])) if np.any(valid_depth) else None,
        "depth_max_m": float(np.max(depth_array[valid_depth])) if np.any(valid_depth) else None,
    }
    if target_visible is not None:
        result["target_visible"] = bool(target_visible)
    return result


def aggregate_camera_quality(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-frame quality metrics without hiding failed frames."""

    if not frames:
        return {"frame_count": 0, "target_visible_fraction": None}
    overexposed = np.asarray([float(f["overexposed_fraction"]) for f in frames])
    depth_valid = np.asarray([float(f["depth_valid_fraction"]) for f in frames])
    underexposed = np.asarray([float(f["underexposed_fraction"]) for f in frames])
    visible = [f["target_visible"] for f in frames if "target_visible" in f]
    return {
        "frame_count": len(frames),
        "overexposed_fraction_mean": float(np.mean(overexposed)),
        "overexposed_fraction_max": float(np.max(overexposed)),
        "underexposed_fraction_mean": float(np.mean(underexposed)),
        "depth_valid_fraction_mean": float(np.mean(depth_valid)),
        "target_visible_fraction": (
            float(np.mean(np.asarray(visible, dtype=bool))) if visible else None
        ),
        "target_visible_frame_count": int(np.count_nonzero(visible)) if visible else None,
    }


def summarize_roi(rgb: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Measure a specified object, retaining an empty mask as missing data.

    Thresholds describe uint8 LDR output, not radiometric sensor response.
    Black space and other objects never enter these statistics.
    """
    rgb = np.asarray(rgb)[..., :3]
    mask = np.asarray(mask, dtype=bool)
    if rgb.shape[:2] != mask.shape or rgb.ndim != 3:
        raise ValueError("ROI mask must match the RGB image dimensions")
    pixels = rgb[mask].astype(float)
    count = len(pixels)
    result = {"pixel_count": count, "underexposed_fraction": None,
              "overexposed_fraction": None, "luma_p05_u8": None,
              "luma_p50_u8": None, "luma_p95_u8": None,
              "bbox_xyxy": None, "edge_margin_px": None}
    if count:
        luma = pixels @ np.array([0.2126, 0.7152, 0.0722])
        y, x = np.nonzero(mask)
        result.update(underexposed_fraction=float(np.mean(luma <= 5.)),
                      overexposed_fraction=float(np.mean(np.any(pixels >= 250., axis=1))),
                      luma_p05_u8=float(np.percentile(luma, 5)),
                      luma_p50_u8=float(np.percentile(luma, 50)),
                      luma_p95_u8=float(np.percentile(luma, 95)),
                      bbox_xyxy=[int(x.min()), int(y.min()), int(x.max()), int(y.max())],
                      edge_margin_px=int(min(x.min(), y.min(), mask.shape[1]-1-x.max(),
                                             mask.shape[0]-1-y.max())))
    return result
