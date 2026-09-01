from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import trimesh
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from pipeline.models import GeometryResult, MeshProfile, MeshResult, PipelineConfig, RunPaths
from pipeline.state import atomic_write_json, sha256_file


_TEMPLATE_ROOT = Path(__file__).resolve().parent.parent / "openfoam" / "templates"
_PROFILE_FACTORS = {
    MeshProfile.COARSE: math.sqrt(2.0),
    MeshProfile.MEDIUM: 1.0,
    MeshProfile.FINE: 1.0 / math.sqrt(2.0),
}
_PROFILE_CELL_LIMIT_KEYS = {
    MeshProfile.COARSE: "max_global_cells_coarse",
    MeshProfile.MEDIUM: "max_global_cells_medium",
    MeshProfile.FINE: "max_global_cells_fine",
}
_TEMPLATE_TARGETS = {
    "0/U.j2": "0/U",
    "0/p.j2": "0/p",
    "0/k.j2": "0/k",
    "0/omega.j2": "0/omega",
    "0/nut.j2": "0/nut",
    "constant/transportProperties.j2": "constant/transportProperties",
    "constant/turbulenceProperties.j2": "constant/turbulenceProperties",
    "system/blockMeshDict.j2": "system/blockMeshDict",
    "system/surfaceFeatureExtractDict.j2": "system/surfaceFeatureExtractDict",
    "system/snappyHexMeshDict.j2": "system/snappyHexMeshDict",
    "system/decomposeParDict.j2": "system/decomposeParDict",
    "system/controlDict.j2": "system/controlDict",
    "system/forces.j2": "system/forces",
}
_STATIC_TARGETS = {
    "system/meshQualityDict": "system/meshQualityDict",
    "system/fvSchemes": "system/fvSchemes",
    "system/fvSolution": "system/fvSolution",
}


def case_rendering_digest() -> str:
    project_root = Path(__file__).resolve().parent.parent
    entries = sorted(path for path in _TEMPLATE_ROOT.rglob("*") if path.is_file())
    entries.append(Path(__file__).resolve())
    digest = hashlib.sha256()
    for path in entries:
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MeshSizing:
    profile: MeshProfile
    domain_min_m: tuple[float, float, float]
    domain_max_m: tuple[float, float, float]
    background_cells: tuple[int, int, int]
    first_layer_thickness_m: float
    max_global_cells: int
    location_in_mesh_m: tuple[float, float, float]
    blockage_fraction: float


@dataclass(frozen=True)
class CaseContext:
    speed_m_s: float
    reynolds_number: float
    turbulent_k_m2_s2: float
    turbulent_omega_s1: float
    density_kg_m3: float
    kinematic_viscosity_m2_s: float
    reference_length_m: float
    reference_area_m2: float
    centre_of_rotation_m: tuple[float, float, float]


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _normalized_bounds(geometry: GeometryResult) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not geometry.normalized_stl.is_file():
        raise ValueError(f"validated normalized STL is unavailable: {geometry.normalized_stl}")
    loaded = trimesh.load(geometry.normalized_stl, process=False, maintain_order=True)
    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        if len(meshes) != 1 or not isinstance(meshes[0], trimesh.Trimesh):
            raise ValueError("validated normalized STL must contain exactly one triangulated hull")
        surface = meshes[0]
    elif isinstance(loaded, trimesh.Trimesh):
        surface = loaded
    else:
        raise ValueError("validated normalized STL is not a triangulated surface")
    bounds = surface.bounds
    if bounds.shape != (2, 3) or not all(math.isfinite(float(value)) for value in bounds.flat):
        raise ValueError("validated normalized STL has non-finite bounds")
    lower = tuple(float(value) for value in bounds[0])
    upper = tuple(float(value) for value in bounds[1])
    if any(high <= low for low, high in zip(lower, upper, strict=True)):
        raise ValueError("validated normalized STL has non-positive bounds")
    return lower, upper


def _farfield_cell_size_m(geometry: GeometryResult, config: PipelineConfig, profile: MeshProfile) -> float:
    mesh = config.raw["mesh"]
    medium = _positive(mesh["medium_farfield_cell_fraction"], "mesh.medium_farfield_cell_fraction") * _positive(
        geometry.length_m, "geometry.length_m"
    )
    return medium * _PROFILE_FACTORS[profile]


def _first_layer_thickness_m(geometry: GeometryResult, config: PipelineConfig) -> float:
    minimum_speed = min(_positive(value, "operating.speeds_m_s") for value in config.raw["operating"]["speeds_m_s"])
    length = _positive(geometry.length_m, "geometry.length_m")
    nu = _positive(config.raw["fluid"]["kinematic_viscosity_m2_s"], "fluid.kinematic_viscosity_m2_s")
    target_yplus = _positive(config.raw["mesh"]["target_yplus_at_min_speed"], "mesh.target_yplus_at_min_speed")
    reynolds = minimum_speed * length / nu
    cf = 0.026 / reynolds ** (1.0 / 7.0)
    friction_velocity = minimum_speed * math.sqrt(cf / 2.0)
    thickness = 2.0 * target_yplus * nu / friction_velocity
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("calculated first-layer thickness is invalid")
    return thickness


def _final_layer_controls(sizing: MeshSizing, config: PipelineConfig) -> tuple[float, float, float]:
    mesh = config.raw["mesh"]
    layers = mesh["surface_layers"]
    surface_level = mesh["surface_level_min"]
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 2:
        raise ValueError("mesh.surface_layers must be an integer of at least 2")
    if isinstance(surface_level, bool) or not isinstance(surface_level, int) or surface_level < 0:
        raise ValueError("mesh.surface_level_min must be a non-negative integer")
    maximum_growth = _positive(mesh["maximum_layer_growth_ratio"], "mesh.maximum_layer_growth_ratio")
    if not 1.0 < maximum_growth < 1.2:
        raise ValueError("mesh.maximum_layer_growth_ratio must be greater than 1 and less than 1.2")
    maximum_fraction = _positive(
        mesh["maximum_final_layer_thickness_fraction"],
        "mesh.maximum_final_layer_thickness_fraction",
    )
    if maximum_fraction > 1.0:
        raise ValueError("mesh.maximum_final_layer_thickness_fraction must not exceed 1")
    level_zero_edges = tuple(
        (high - low) / count
        for low, high, count in zip(
            sizing.domain_min_m,
            sizing.domain_max_m,
            sizing.background_cells,
            strict=True,
        )
    )
    coarsest_surface_cell = min(level_zero_edges) / (2**surface_level)
    desired_final_layer = sizing.first_layer_thickness_m * maximum_growth ** (layers - 1)
    final_fraction = min(maximum_fraction, desired_final_layer / coarsest_surface_cell)
    if not math.isfinite(final_fraction) or final_fraction <= 0.0:
        raise ValueError("calculated final-layer thickness fraction is invalid")
    return final_fraction, coarsest_surface_cell, desired_final_layer


def calculate_mesh_sizing(geometry: GeometryResult, config: PipelineConfig, profile: MeshProfile) -> MeshSizing:
    try:
        profile = MeshProfile(profile)
    except ValueError as error:
        raise ValueError(f"unsupported mesh profile: {profile!r}") from error
    hull_min, hull_max = _normalized_bounds(geometry)
    length = _positive(geometry.length_m, "geometry.length_m")
    area = _positive(geometry.frontal_area_m2, "geometry.frontal_area_m2")
    domain = config.raw["domain"]
    upstream = _positive(domain["upstream_lengths"], "domain.upstream_lengths") * length
    downstream = _positive(domain["downstream_lengths"], "domain.downstream_lengths") * length
    radial = _positive(domain["radial_lengths"], "domain.radial_lengths") * length
    lower = (hull_min[0] - upstream, hull_min[1] - radial, hull_min[2] - radial)
    upper = (hull_max[0] + downstream, hull_max[1] + radial, hull_max[2] + radial)
    farfield = _farfield_cell_size_m(geometry, config, profile)
    cells = tuple(max(8, math.ceil((high - low) / farfield)) for low, high in zip(lower, upper, strict=True))
    cross_section = (upper[1] - lower[1]) * (upper[2] - lower[2])
    blockage = area / cross_section
    maximum_blockage = _positive(domain["maximum_blockage"], "domain.maximum_blockage")
    if blockage > maximum_blockage:
        raise ValueError(
            f"frontal blockage {blockage:.6%} exceeds configured limit {maximum_blockage:.6%}"
        )
    location = (0.0, 4.0 * length, 4.0 * length)
    if not all(low < coordinate < high for coordinate, low, high in zip(location, lower, upper, strict=True)):
        raise ValueError("locationInMesh (0, 4L, 4L) is outside the calculated domain")
    if all(low <= coordinate <= high for coordinate, low, high in zip(location, hull_min, hull_max, strict=True)):
        raise ValueError("locationInMesh (0, 4L, 4L) is not proven outside the hull bounds")
    limit_raw = config.raw["mesh"].get(_PROFILE_CELL_LIMIT_KEYS[profile])
    if isinstance(limit_raw, bool) or not isinstance(limit_raw, int) or limit_raw <= 0:
        raise ValueError(f"mesh.{_PROFILE_CELL_LIMIT_KEYS[profile]} must be a positive integer")
    return MeshSizing(
        profile=profile,
        domain_min_m=lower,
        domain_max_m=upper,
        background_cells=cells,
        first_layer_thickness_m=_first_layer_thickness_m(geometry, config),
        max_global_cells=limit_raw,
        location_in_mesh_m=location,
        blockage_fraction=blockage,
    )


def calculate_turbulence(speed_m_s: float, geometry: GeometryResult, config: PipelineConfig) -> tuple[float, float]:
    speed = _positive(speed_m_s, "speed_m_s")
    intensity = _positive(config.raw["operating"]["turbulence_intensity"], "operating.turbulence_intensity")
    fraction = _positive(
        config.raw["operating"]["turbulence_length_fraction"], "operating.turbulence_length_fraction"
    )
    transverse = max(_positive(value, "geometry.normalized_extents_m") for value in geometry.normalized_extents_m[1:])
    turbulent_k = 1.5 * (speed * intensity) ** 2
    length_scale = fraction * transverse
    turbulent_omega = math.sqrt(turbulent_k) / (0.09**0.25 * length_scale)
    if not all(math.isfinite(value) and value > 0.0 for value in (turbulent_k, turbulent_omega)):
        raise ValueError("calculated turbulence quantities are invalid")
    return turbulent_k, turbulent_omega


def _case_context(speed_m_s: float, geometry: GeometryResult, config: PipelineConfig) -> CaseContext:
    speed = _positive(speed_m_s, "speed_m_s")
    length = _positive(geometry.length_m, "geometry.length_m")
    nu = _positive(config.raw["fluid"]["kinematic_viscosity_m2_s"], "fluid.kinematic_viscosity_m2_s")
    k, omega = calculate_turbulence(speed, geometry, config)
    centre = tuple(float(value) for value in geometry.centroid_m)
    if len(centre) != 3 or not all(math.isfinite(value) for value in centre):
        raise ValueError("geometry.centroid_m must be three finite coordinates")
    return CaseContext(
        speed_m_s=speed,
        reynolds_number=speed * length / nu,
        turbulent_k_m2_s2=k,
        turbulent_omega_s1=omega,
        density_kg_m3=_positive(config.raw["fluid"]["density_kg_m3"], "fluid.density_kg_m3"),
        kinematic_viscosity_m2_s=nu,
        reference_length_m=length,
        reference_area_m2=_positive(geometry.frontal_area_m2, "geometry.frontal_area_m2"),
        centre_of_rotation_m=centre,
    )


def _foam_scalar(value: float | int) -> str:
    return f"{value:.12g}"


def _foam_vector(value: tuple[float, float, float]) -> str:
    return "(" + " ".join(_foam_scalar(component) for component in value) + ")"


def _render_values(geometry: GeometryResult, sizing: MeshSizing, context: CaseContext, config: PipelineConfig) -> dict[str, Any]:
    first_layer = sizing.first_layer_thickness_m
    final_layer_fraction, coarsest_surface_cell, desired_final_layer = _final_layer_controls(sizing, config)
    hull_min, hull_max = _normalized_bounds(geometry)
    length = context.reference_length_m
    transverse = max(hull_max[index] - hull_min[index] for index in (1, 2))
    axial_padding = _positive(
        config.raw["mesh"]["nearbody_streamwise_padding_lengths"],
        "mesh.nearbody_streamwise_padding_lengths",
    ) * length
    transverse_padding = _positive(
        config.raw["mesh"]["nearbody_transverse_padding_diameters"],
        "mesh.nearbody_transverse_padding_diameters",
    ) * transverse
    nearbody_min = (
        hull_min[0] - axial_padding,
        hull_min[1] - transverse_padding,
        hull_min[2] - transverse_padding,
    )
    nearbody_max = (
        hull_max[0] + axial_padding,
        hull_max[1] + transverse_padding,
        hull_max[2] + transverse_padding,
    )
    wake_start = hull_min[0] + float(config.raw["mesh"]["wake_start_fraction"]) * length
    wake_half_width = _positive(
        config.raw["mesh"]["wake_half_width_diameters"],
        "mesh.wake_half_width_diameters",
    ) * transverse
    wake_length = _positive(config.raw["mesh"]["wake_lengths"], "mesh.wake_lengths") * length
    flow_time = length / context.speed_m_s
    solver = config.raw["solver"]
    time_step = _positive(solver["time_step_flow_fraction"], "solver.time_step_flow_fraction") * flow_time
    field_average_window_time = (
        _positive(solver["field_average_window_flow_times"], "solver.field_average_window_flow_times")
        * flow_time
    )
    first_checkpoint_time = (
        _positive(solver["first_convergence_check_flow_times"], "solver.first_convergence_check_flow_times")
        * flow_time
    )
    checkpoint_time = _positive(solver["checkpoint_flow_times"], "solver.checkpoint_flow_times") * flow_time
    return {
        "speed_m_s": _foam_scalar(context.speed_m_s),
        "speed_vector": _foam_vector((context.speed_m_s, 0.0, 0.0)),
        "reynolds_number": _foam_scalar(context.reynolds_number),
        "turbulent_k_m2_s2": _foam_scalar(context.turbulent_k_m2_s2),
        "turbulent_omega_s1": _foam_scalar(context.turbulent_omega_s1),
        "density_kg_m3": _foam_scalar(context.density_kg_m3),
        "kinematic_viscosity_m2_s": _foam_scalar(context.kinematic_viscosity_m2_s),
        "reference_length_m": _foam_scalar(context.reference_length_m),
        "reference_area_m2": _foam_scalar(context.reference_area_m2),
        "centre_of_rotation_m": _foam_vector(context.centre_of_rotation_m),
        "domain_min_m": _foam_vector(sizing.domain_min_m),
        "domain_max_m": _foam_vector(sizing.domain_max_m),
        "background_cells": "(" + " ".join(str(value) for value in sizing.background_cells) + ")",
        "first_layer_thickness_m": _foam_scalar(first_layer),
        "minimum_layer_thickness_m": _foam_scalar(first_layer),
        "final_layer_thickness_fraction": _foam_scalar(final_layer_fraction),
        "coarsest_surface_cell_size_m": _foam_scalar(coarsest_surface_cell),
        "desired_final_layer_thickness_m": _foam_scalar(desired_final_layer),
        "maximum_layer_growth_ratio": _foam_scalar(float(config.raw["mesh"]["maximum_layer_growth_ratio"])),
        "farfield_cell_size_m": _foam_scalar(_farfield_cell_size_m(geometry, config, sizing.profile)),
        "max_global_cells": sizing.max_global_cells,
        "location_in_mesh_m": _foam_vector(sizing.location_in_mesh_m),
        "nearbody_min_m": _foam_vector(nearbody_min),
        "nearbody_max_m": _foam_vector(nearbody_max),
        "wake_min_m": _foam_vector((wake_start, -wake_half_width, -wake_half_width)),
        "wake_max_m": _foam_vector((
            hull_max[0] + wake_length,
            wake_half_width,
            wake_half_width,
        )),
        "surface_level_min": int(config.raw["mesh"]["surface_level_min"]),
        "surface_level_max": int(config.raw["mesh"]["surface_level_max"]),
        "nearbody_level": int(config.raw["mesh"]["nearbody_level"]),
        "wake_level": int(config.raw["mesh"]["wake_level"]),
        "surface_layers": int(config.raw["mesh"]["surface_layers"]),
        "maximum_non_orthogonality": _foam_scalar(float(config.raw["mesh"]["max_non_orthogonality"])),
        "maximum_internal_skewness": _foam_scalar(float(config.raw["mesh"]["max_internal_skewness"])),
        "minimum_determinant": _foam_scalar(float(config.raw["mesh"]["minimum_determinant"])),
        "mpi_ranks": int(config.raw["mpi_ranks"]),
        "flow_time_s": _foam_scalar(flow_time),
        "time_step_s": _foam_scalar(time_step),
        "target_courant_number": _foam_scalar(
            _positive(solver["target_courant_number"], "solver.target_courant_number")
        ),
        "averaging_window_time_s": _foam_scalar(field_average_window_time),
        "first_checkpoint_time_s": _foam_scalar(first_checkpoint_time),
        "maximum_time_s": _foam_scalar(_positive(solver["maximum_flow_times"], "solver.maximum_flow_times") * flow_time),
        "write_interval_s": _foam_scalar(checkpoint_time),
        "force_write_interval": int(config.raw["solver"]["force_write_interval"]),
        "drag_direction": "(1 0 0)",
        "lift_direction": "(0 0 1)",
        "pitch_axis": "(0 1 0)",
    }


def _render_case(
    case_dir: Path,
    geometry: GeometryResult,
    sizing: MeshSizing,
    context: CaseContext,
    config: PipelineConfig,
    *,
    include_tri_surface: bool = True,
) -> None:
    environment = Environment(loader=FileSystemLoader(_TEMPLATE_ROOT), undefined=StrictUndefined, keep_trailing_newline=True)
    values = _render_values(geometry, sizing, context, config)
    for source, target in _TEMPLATE_TARGETS.items():
        output = case_dir / target
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(environment.get_template(source).render(**values), encoding="utf-8")
    for source, target in _STATIC_TARGETS.items():
        output = case_dir / target
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_TEMPLATE_ROOT / source, output)
    if include_tri_surface:
        tri_surface = case_dir / "constant" / "triSurface"
        tri_surface.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(geometry.normalized_stl, tri_surface / "hull.stl")
    speed_contexts = {
        f"{float(speed):.2f}": {
            **asdict(_case_context(float(speed), geometry, config)),
            "drag_direction": [1.0, 0.0, 0.0],
            "lift_direction": [0.0, 0.0, 1.0],
            "pitch_axis": [0.0, 1.0, 0.0],
        }
        for speed in config.raw["operating"]["speeds_m_s"]
    }
    atomic_write_json(
        case_dir / "case-context.json",
        {
            "schema_version": 2,
            "mesh_profile": sizing.profile.value,
            "mesh_sizing": asdict(sizing),
            "reference_context": asdict(context),
            "speed_contexts": speed_contexts,
            "rendered_scalars": values,
        },
    )


def render_mesh_case(paths: RunPaths, geometry: GeometryResult, sizing: MeshSizing, config: PipelineConfig) -> Path:
    case_dir = paths.cases / sizing.profile.value
    if case_dir.exists():
        raise ValueError(f"prepared mesh case already exists and is preserved: {case_dir}")
    case_dir.mkdir(parents=True)
    _render_case(
        case_dir,
        geometry,
        sizing,
        _case_context(float(config.raw["operating"]["reference_speed_m_s"]), geometry, config),
        config,
    )
    return case_dir


def render_solver_case(paths: RunPaths, geometry: GeometryResult, mesh: MeshResult, speed_m_s: float, config: PipelineConfig) -> Path:
    source = mesh.case_dir.resolve()
    if not source.is_dir():
        raise ValueError(f"mesh case directory does not exist: {source}")
    profile = MeshProfile(mesh.profile)
    speed_root = paths.cases / profile.value / "speeds"
    base_name = f"{_positive(speed_m_s, 'speed_m_s'):.2f}"
    destination = speed_root / base_name
    attempt = 2
    while destination.exists():
        destination = speed_root / f"{base_name}-attempt-{attempt:02d}"
        attempt += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    _render_case(
        destination,
        geometry,
        calculate_mesh_sizing(geometry, config, profile),
        _case_context(speed_m_s, geometry, config),
        config,
        include_tri_surface=False,
    )
    return destination


def write_task_checkpoint(paths: RunPaths) -> Path:
    project_root = paths.root.parent.parent
    evidence = project_root / "docs" / "superpowers" / "plans" / "evidence" / "task-03.sha256"
    entries = [*_TEMPLATE_ROOT.rglob("*"), Path(__file__).resolve()]
    entries = sorted(path for path in entries if path.is_file())
    entries.extend(paths.cases / profile.value / "case-context.json" for profile in MeshProfile)
    missing = [path for path in entries if not path.is_file()]
    if missing:
        raise ValueError("cannot write Task 3 checkpoint; missing: " + ", ".join(str(path) for path in missing))
    lines = [f"{sha256_file(path)}  {path.relative_to(project_root)}" for path in entries]
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return evidence
