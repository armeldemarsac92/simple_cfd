from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage

from pipeline.models import CaseResult, GeometryResult, VisualizationResult
from pipeline.postprocess import WallSurfaceData, read_wall_surface
from pipeline.runtime import require_executable, run_checked
from pipeline.state import atomic_write_json, sha256_file


_PLANE_FUNCTION = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object auvLongitudinalPlane;
}

type surfaces;
libs (sampling);
writeControl writeTime;
surfaceFormat vtk;
fields (pMean UMean);
interpolationScheme cellPoint;

formatOptions
{
    vtk
    {
        format ascii;
        legacy true;
    }
}

surfaces
{
    centrePlane
    {
        type plane;
        point (0 0 0);
        normal (0 1 0);
        triangulate true;
        interpolate true;
    }
}
"""


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> Path:
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, destination)
    return destination.resolve(strict=True)


def _save_figure(figure: plt.Figure, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    figure.savefig(temporary, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)
    return destination.resolve(strict=True)


def _matching_field(fields: Iterable[str], requested: str) -> str:
    candidates = [name for name in fields if name == requested or name.rsplit(":", 1)[-1] == requested]
    if len(candidates) != 1:
        raise ValueError(f"expected one VTK field named {requested!r}, found {candidates}")
    return candidates[0]


def _density(case_dir: Path) -> float:
    context_path = case_dir / "case-context.json"
    payload = json.loads(context_path.read_text(encoding="utf-8"))
    context = payload.get("reference_context")
    if not isinstance(context, dict):
        raise ValueError(f"case context lacks reference_context: {context_path}")
    density = float(context["density_kg_m3"])
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError(f"case context contains invalid density: {context_path}")
    return density


def _cell_scalar(surface: WallSurfaceData, name: str) -> np.ndarray:
    field_name = _matching_field(surface.cell_fields, name)
    values = np.asarray(surface.cell_fields[field_name], dtype=float).reshape(-1)
    if len(values) != len(surface.cells) or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid cell scalar field {name!r}")
    return values


def _cell_vector(surface: WallSurfaceData, name: str) -> np.ndarray:
    field_name = _matching_field(surface.cell_fields, name)
    values = np.asarray(surface.cell_fields[field_name], dtype=float)
    if values.shape != (len(surface.cells), 3) or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid cell vector field {name!r}")
    return values


def _hull_norm(values: np.ndarray, kind: str) -> tuple[Normalize, str, str]:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size == 0:
        raise ValueError(f"empty finite values for {kind}")
    if kind == "cp":
        bound = max(abs(float(np.quantile(finite, 0.01))), abs(float(np.quantile(finite, 0.99))), 1.0e-9)
        return TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound), "coolwarm", "Pressure coefficient Cp"
    if kind == "wall_shear":
        upper = max(float(np.quantile(finite, 0.99)), 1.0e-12)
        return Normalize(vmin=0.0, vmax=upper), "magma", "Wall shear magnitude (Pa)"
    if kind == "yplus":
        lower = max(float(np.quantile(finite, 0.01)), 1.0)
        upper = max(float(np.quantile(finite, 0.99)), lower * 1.001)
        return LogNorm(vmin=lower, vmax=upper), "viridis", "y+ (log scale)"
    raise ValueError(f"unknown hull field kind: {kind}")


def _render_hull_field(
    surface: WallSurfaceData,
    values: np.ndarray,
    kind: str,
    speed_m_s: float,
    destination: Path,
) -> Path:
    norm, cmap_name, label = _hull_norm(values, kind)
    cmap = matplotlib.colormaps[cmap_name]
    polygons = [surface.points[np.asarray(cell, dtype=int)] for cell in surface.cells]
    figure = plt.figure(figsize=(12.8, 7.2), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    collection = Poly3DCollection(
        polygons,
        facecolors=cmap(norm(values)),
        edgecolors="none",
        linewidths=0.0,
        antialiased=False,
        rasterized=True,
    )
    axis.add_collection3d(collection, autolim=False)
    minimum = np.min(surface.points, axis=0)
    maximum = np.max(surface.points, axis=0)
    extent = maximum - minimum
    margin = np.maximum(0.04 * extent, 1.0e-4)
    axis.set_xlim(minimum[0] - margin[0], maximum[0] + margin[0], view_margin=0.0)
    axis.set_ylim(minimum[1] - margin[1], maximum[1] + margin[1], view_margin=0.0)
    axis.set_zlim(minimum[2] - margin[2], maximum[2] + margin[2], view_margin=0.0)
    axis.set_autoscale_on(False)
    axis.set_box_aspect(tuple(float(max(value, 1.0e-6)) for value in extent))
    axis.set_proj_type("ortho")
    axis.view_init(elev=22, azim=-118)
    axis.set_xlabel("X — relative flow (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title(f"AUV hull — {label} — U∞ = {speed_m_s:.2f} m/s", pad=18)
    arrow_start = minimum + np.asarray((0.08 * extent[0], 0.92 * extent[1], 0.90 * extent[2]))
    axis.quiver(
        arrow_start[0],
        arrow_start[1],
        arrow_start[2],
        0.32 * extent[0],
        0.0,
        0.0,
        color="#1f2937",
        linewidth=2.0,
        arrow_length_ratio=0.12,
    )
    axis.text(
        arrow_start[0] + 0.16 * extent[0],
        arrow_start[1],
        arrow_start[2],
        "flow +X",
        color="#1f2937",
        ha="center",
    )
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array(values)
    colorbar = figure.colorbar(scalar, ax=axis, shrink=0.72, pad=0.08)
    colorbar.set_label(label)
    axis.text2D(
        0.02,
        0.02,
        f"finite range: {float(np.min(values)):.5g} to {float(np.max(values)):.5g}",
        transform=axis.transAxes,
        fontsize=9,
        color="#374151",
    )
    return _save_figure(figure, destination)


def _plane_dictionary(case_dir: Path) -> Path:
    destination = case_dir / "system" / "auvLongitudinalPlane"
    _atomic_text(destination, _PLANE_FUNCTION)
    return destination


def _sample_plane(case_dir: Path, destination: Path) -> tuple[Path, Path, Path]:
    function_dictionary = _plane_dictionary(case_dir)
    log_path = destination / "log.sampleLongitudinalPlane"
    run_checked(
        [
            require_executable("pimpleFoam"),
            "-case",
            case_dir,
            "-postProcess",
            "-latestTime",
            "-func",
            function_dictionary.stem,
        ],
        case_dir,
        log_path,
    )
    candidates = sorted(
        (case_dir / "postProcessing" / function_dictionary.stem).rglob("centrePlane.vtk"),
        key=lambda path: (float(path.parent.name), path.stat().st_mtime_ns),
    )
    if not candidates:
        raise ValueError(f"OpenFOAM did not write centrePlane.vtk; log: {log_path}")
    raw_plane = _atomic_copy(candidates[-1], destination / "raw" / "longitudinal-plane.vtk")
    return raw_plane, function_dictionary.resolve(strict=True), log_path.resolve(strict=True)


def _fan_triangles(cells: Iterable[np.ndarray]) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    for cell in cells:
        indices = np.asarray(cell, dtype=int)
        for index in range(1, len(indices) - 1):
            triangles.append((int(indices[0]), int(indices[index]), int(indices[index + 1])))
    if not triangles:
        raise ValueError("sampled plane contains no polygon triangles")
    return np.asarray(triangles, dtype=int)


def _plane_field(surface: WallSurfaceData, name: str, components: int) -> np.ndarray:
    field_name = _matching_field(surface.point_fields, name)
    values = np.asarray(surface.point_fields[field_name], dtype=float)
    expected = (len(surface.points), components)
    if values.shape != expected or not np.all(np.isfinite(values)):
        raise ValueError(f"sampled plane field {name!r} has shape {values.shape}, expected {expected}")
    return values


def _masked_interpolation(
    triangulation: mtri.Triangulation,
    values: np.ndarray,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
) -> np.ma.MaskedArray:
    interpolator = mtri.LinearTriInterpolator(triangulation, values)
    interpolated = np.ma.asarray(interpolator(x_grid, z_grid), dtype=float)
    return np.ma.masked_invalid(interpolated)


def _fill_small_cut_holes(
    grids: tuple[np.ma.MaskedArray, ...],
    mask: np.ndarray,
) -> tuple[tuple[np.ma.MaskedArray, ...], np.ndarray, int]:
    labels, component_count = ndimage.label(mask)
    if component_count == 0:
        return grids, mask, 0
    border_labels = set(
        int(value)
        for value in np.unique(
            np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
        )
    )
    sizes = np.bincount(labels.reshape(-1))
    maximum_small_hole = max(64, int(mask.size * 0.003))
    selected = [
        label
        for label in range(1, component_count + 1)
        if label not in border_labels and int(sizes[label]) <= maximum_small_hole
    ]
    if not selected:
        return grids, mask, 0
    fill_mask = np.isin(labels, selected)
    _, nearest = ndimage.distance_transform_edt(mask, return_indices=True)
    remaining_mask = mask & ~fill_mask
    filled: list[np.ma.MaskedArray] = []
    for grid in grids:
        values = np.asarray(np.ma.asarray(grid).filled(np.nan), dtype=float)
        values[fill_mask] = values[tuple(index[fill_mask] for index in nearest)]
        filled.append(np.ma.array(values, mask=remaining_mask))
    return tuple(filled), remaining_mask, len(selected)


def _render_plane(
    plane: WallSurfaceData,
    hull: WallSurfaceData,
    speed_m_s: float,
    length_m: float,
    destination: Path,
) -> tuple[Path, int, int]:
    if speed_m_s <= 0.0 or length_m <= 0.0:
        raise ValueError("positive speed and reference length are required for normalized visualization")
    velocity = _plane_field(plane, "UMean", 3)
    pressure = _plane_field(plane, "pMean", 1).reshape(-1)
    x = plane.points[:, 0] / length_m
    z = plane.points[:, 2] / length_m
    triangles = _fan_triangles(plane.cells)
    first = np.column_stack((x[triangles[:, 1]] - x[triangles[:, 0]], z[triangles[:, 1]] - z[triangles[:, 0]]))
    second = np.column_stack((x[triangles[:, 2]] - x[triangles[:, 0]], z[triangles[:, 2]] - z[triangles[:, 0]]))
    signed_double_area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    finite_area = np.abs(signed_double_area) > 1.0e-14
    positive_count = int(np.count_nonzero(signed_double_area[finite_area] > 0.0))
    negative_count = int(np.count_nonzero(signed_double_area[finite_area] < 0.0))
    dominant_sign = 1.0 if positive_count >= negative_count else -1.0
    consistent = finite_area & (signed_double_area * dominant_sign > 0.0)
    discarded_triangles = int(len(triangles) - np.count_nonzero(consistent))
    triangles = np.array(triangles[consistent], copy=True)
    signed_double_area = signed_double_area[consistent]
    clockwise = signed_double_area < 0.0
    triangles[clockwise, 1], triangles[clockwise, 2] = (
        np.array(triangles[clockwise, 2], copy=True),
        np.array(triangles[clockwise, 1], copy=True),
    )
    triangulation = mtri.Triangulation(x, z, triangles)

    hull_min = np.min(hull.points, axis=0)
    hull_max = np.max(hull.points, axis=0)
    x_limits = (hull_min[0] / length_m - 0.50, hull_max[0] / length_m + 2.00)
    z_half = max(0.75, 3.0 * max(abs(hull_min[2]), abs(hull_max[2])) / length_m)
    z_limits = (-z_half, z_half)
    x_axis = np.linspace(*x_limits, 620)
    z_axis = np.linspace(*z_limits, 300)
    x_grid, z_grid = np.meshgrid(x_axis, z_axis)

    normalized_speed = np.linalg.norm(velocity, axis=1) / speed_m_s
    pressure_coefficient = 2.0 * pressure / speed_m_s**2
    normalized_ux = velocity[:, 0] / speed_m_s
    normalized_uz = velocity[:, 2] / speed_m_s
    speed_grid = _masked_interpolation(triangulation, normalized_speed, x_grid, z_grid)
    pressure_grid = _masked_interpolation(triangulation, pressure_coefficient, x_grid, z_grid)
    ux_grid = _masked_interpolation(triangulation, normalized_ux, x_grid, z_grid)
    uz_grid = _masked_interpolation(triangulation, normalized_uz, x_grid, z_grid)
    combined_mask = (
        np.ma.getmaskarray(speed_grid)
        | np.ma.getmaskarray(pressure_grid)
        | np.ma.getmaskarray(ux_grid)
        | np.ma.getmaskarray(uz_grid)
    )
    speed_grid = np.ma.array(speed_grid, mask=combined_mask)
    pressure_grid = np.ma.array(pressure_grid, mask=combined_mask)
    ux_grid = np.ma.array(ux_grid, mask=combined_mask)
    uz_grid = np.ma.array(uz_grid, mask=combined_mask)
    (speed_grid, pressure_grid, ux_grid, uz_grid), combined_mask, filled_cut_holes = _fill_small_cut_holes(
        (speed_grid, pressure_grid, ux_grid, uz_grid),
        combined_mask,
    )
    finite_speed = speed_grid.compressed()
    finite_pressure = pressure_grid.compressed()
    if finite_speed.size == 0 or finite_pressure.size == 0:
        raise ValueError("sampled plane interpolation produced no finite visualization values")

    speed_upper = max(1.05, float(np.quantile(finite_speed, 0.99)))
    pressure_bound = max(
        abs(float(np.quantile(finite_pressure, 0.01))),
        abs(float(np.quantile(finite_pressure, 0.99))),
        1.0e-6,
    )
    figure, axes = plt.subplots(2, 1, figsize=(14.0, 8.4), sharex=True, constrained_layout=True)
    speed_levels = np.linspace(0.0, speed_upper, 48)
    speed_contours = axes[0].contourf(
        x_grid,
        z_grid,
        speed_grid,
        levels=speed_levels,
        cmap="viridis",
        extend="max",
    )
    pressure_levels = np.linspace(-pressure_bound, pressure_bound, 49)
    pressure_contours = axes[1].contourf(
        x_grid,
        z_grid,
        pressure_grid,
        levels=pressure_levels,
        cmap="coolwarm",
        extend="both",
    )
    for axis in axes:
        axis.streamplot(
            x_axis,
            z_axis,
            ux_grid,
            uz_grid,
            color="#111827",
            density=1.15,
            linewidth=0.45,
            arrowsize=0.65,
            minlength=0.08,
        )
        valid = (~combined_mask).astype(float)
        axis.contour(x_grid, z_grid, valid, levels=(0.5,), colors=("#111827",), linewidths=(0.8,))
        axis.axvline(hull_min[0] / length_m, color="#6b7280", linestyle="--", linewidth=0.7)
        axis.axvline(hull_max[0] / length_m, color="#6b7280", linestyle="--", linewidth=0.7)
        axis.set_ylim(*z_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_ylabel("z / L")
        axis.grid(color="#9ca3af", alpha=0.15, linewidth=0.4)
    axes[0].set_title(f"Time-averaged longitudinal plane — normalized speed and streamlines — U∞ = {speed_m_s:.2f} m/s")
    axes[1].set_title("Time-averaged kinematic-pressure coefficient Cp = 2p̄/U∞²")
    axes[1].set_xlabel("x / L   (relative flow +X)")
    figure.colorbar(speed_contours, ax=axes[0], label="|U| / U∞", pad=0.015)
    figure.colorbar(pressure_contours, ax=axes[1], label="Cp", pad=0.015)
    return _save_figure(figure, destination), discarded_triangles, filled_cut_holes


def _artifact(path: Path | None) -> dict[str, str] | None:
    if path is None or not path.is_file():
        return None
    return {"path": str(path), "sha256": sha256_file(path)}


def render_case_visuals(
    case: CaseResult,
    geometry: GeometryResult,
    destination: Path,
) -> VisualizationResult:
    """Render non-blocking CFD diagnostics for one accepted final field."""
    case_dir = case.case_dir.resolve(strict=True)
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    hull_vtk: Path | None = None
    plane_vtk: Path | None = None
    case_marker: Path | None = None
    cp_png: Path | None = None
    wall_shear_png: Path | None = None
    yplus_png: Path | None = None
    flow_png: Path | None = None
    function_dictionary: Path | None = None
    sampling_log: Path | None = None

    marker = case_dir / "case.foam"
    try:
        if marker.exists() and marker.stat().st_size != 0:
            warnings.append("case_marker_not_zero_bytes")
        else:
            marker.touch(exist_ok=True)
            case_marker = marker.resolve(strict=True)
    except OSError as error:
        warnings.append(f"case_marker_failed: {error}")

    surface: WallSurfaceData | None = None
    try:
        hull_vtk = _atomic_copy(case.wall_fields_path, output / "raw" / "hull-wall.vtk")
        surface = read_wall_surface(hull_vtk)
        density = _density(case_dir)
        cp = _cell_scalar(surface, "cpMean")
        wall_shear_pa = np.linalg.norm(_cell_vector(surface, "wallShearStressMean"), axis=1) * density
        yplus = _cell_scalar(surface, "yPlus")
        cp_png = _render_hull_field(surface, cp, "cp", case.speed_m_s, output / "hull-cp.png")
        wall_shear_png = _render_hull_field(
            surface,
            wall_shear_pa,
            "wall_shear",
            case.speed_m_s,
            output / "hull-wall-shear.png",
        )
        yplus_png = _render_hull_field(surface, yplus, "yplus", case.speed_m_s, output / "hull-yplus.png")
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        warnings.append(f"hull_visualization_failed: {error}")

    try:
        plane_vtk, function_dictionary, sampling_log = _sample_plane(case_dir, output)
        plane = read_wall_surface(plane_vtk)
        if surface is None:
            surface = read_wall_surface(case.wall_fields_path)
        flow_png, discarded_triangles, filled_cut_holes = _render_plane(
            plane,
            surface,
            case.speed_m_s,
            geometry.length_m,
            output / "longitudinal-flow.png",
        )
        if discarded_triangles:
            warnings.append(f"plane_inconsistent_triangles_discarded: {discarded_triangles}")
        if filled_cut_holes:
            warnings.append(f"plane_cut_holes_interpolated: {filled_cut_holes}")
        if sampling_log is not None and "Failed cuts for" in sampling_log.read_text(
            encoding="utf-8", errors="replace"
        ):
            warnings.append("openfoam_plane_cut_warning")
    except (OSError, RuntimeError, ValueError) as error:
        warnings.append(f"longitudinal_visualization_failed: {error}")

    result = VisualizationResult(
        destination=output,
        hull_vtk=hull_vtk,
        plane_vtk=plane_vtk,
        case_marker=case_marker,
        cp_png=cp_png,
        wall_shear_png=wall_shear_png,
        yplus_png=yplus_png,
        flow_png=flow_png,
        warnings=tuple(warnings),
    )
    atomic_write_json(
        output / "visualization-result.json",
        {
            "schema_version": 2,
            "case_dir": str(case_dir),
            "speed_m_s": case.speed_m_s,
            "mesh_profile": case.mesh_profile.value,
            "complete": all(path is not None for path in (cp_png, wall_shear_png, yplus_png, flow_png)),
            "warnings": list(warnings),
            "artifacts": {
                "hull_vtk": _artifact(hull_vtk),
                "plane_vtk": _artifact(plane_vtk),
                "case_marker": _artifact(case_marker),
                "cp_png": _artifact(cp_png),
                "wall_shear_png": _artifact(wall_shear_png),
                "yplus_png": _artifact(yplus_png),
                "flow_png": _artifact(flow_png),
                "function_dictionary": _artifact(function_dictionary),
                "sampling_log": _artifact(sampling_log),
            },
        },
    )
    return result
