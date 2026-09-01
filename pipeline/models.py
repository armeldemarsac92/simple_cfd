from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Stage(StrEnum):
    DISCOVER = "discover"
    GEOMETRY = "geometry"
    PREPARE = "prepare"
    MESH = "mesh"
    REFERENCE = "reference"
    SWEEP = "sweep"
    PERSIST = "persist"
    REPORT = "report"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class MeshProfile(StrEnum):
    COARSE = "coarse"
    MEDIUM = "medium"
    FINE = "fine"


@dataclass(frozen=True)
class OpenFoamRuntime:
    bashrc: Path
    project_dir: Path
    version: str
    build: str
    mpirun: Path
    pimple_foam: Path


@dataclass(frozen=True)
class RunPaths:
    root: Path
    manifest: Path
    status: Path
    geometry: Path
    meshes: Path
    cases: Path
    postprocessing: Path


@dataclass(frozen=True)
class GeometryResult:
    source: Path
    source_sha256: str
    declared_unit: str
    scale_to_m: float
    normalized_stl: Path
    original_bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    normalized_extents_m: tuple[float, float, float]
    length_m: float
    frontal_area_m2: float
    wetted_area_m2: float
    volume_m3: float
    centroid_m: tuple[float, float, float]
    bow_original_axis: str
    bow_confidence: float
    transform: tuple[tuple[float, float, float, float], ...]
    inverse_transform: tuple[tuple[float, float, float, float], ...]
    original_stl: Path
    orientation_metadata: Path
    preview_png: Path
    surface_check_log: Path
    step_unit_entity: str


@dataclass(frozen=True)
class MeshResult:
    profile: MeshProfile
    case_dir: Path
    mesh_sha256: str
    cell_count: int
    surface_face_count: int
    layer_coverage_fraction: float
    max_non_orthogonality: float
    max_skewness: float
    diagnostic_min_wellposedness_determinant: float
    min_volume: float
    check_mesh_passed: bool
    illegal_faces: int = 0
    disconnected_regions: int = 1
    faces_with_layers: int = 0
    minimum_layer_count: int = 0
    maximum_layer_count: int = 0
    final_surface_deviation_m: float = 0.0
    mesh_ok: bool = False
    all_geometry_passed: bool = False
    concave_cells: int = 0


@dataclass(frozen=True)
class CaseResult:
    case_dir: Path
    speed_m_s: float
    mesh_profile: MeshProfile
    mesh_sha256: str
    reynolds_number: float
    accepted: bool
    rejection_code: str | None
    diagnostic_codes: tuple[str, ...]
    force_total_n: tuple[float, float, float]
    force_pressure_n: tuple[float, float, float]
    force_viscous_n: tuple[float, float, float]
    moment_nm: tuple[float, float, float]
    drag_n: float
    drag_coefficient: float
    tow_power_w: float
    force_stationary_mean_n: float
    force_stationary_cov: float
    force_mean_drift: float
    force_stationary_start_time_s: float
    force_stationary_end_time_s: float
    force_stationary_duration_flow_times: float
    force_mean_uncertainty_n: float
    force_mean_uncertainty_fraction: float
    force_integral_time_scale_s: float
    force_effective_sample_count: float
    yplus: dict[str, float]
    cp_min: float
    minimum_absolute_pressure_pa: float
    cavitation_margin_pa: float
    residuals: dict[str, dict[str, float]]
    physical_time_s: float
    time_steps: int
    maximum_courant_number: float
    force_history_path: Path
    solver_info_path: Path
    wall_fields_path: Path


@dataclass(frozen=True)
class GridStudyResult:
    classification: str
    effective_r21: float
    effective_r32: float
    observed_order: float | None
    extrapolated_drag_n: float | None
    fine_gci_percent: float | None
    coarse_medium_change_percent: float
    medium_fine_change_percent: float
    accepted: bool
    reason: str
    component_details: dict[str, dict[str, float | str | None]]


@dataclass(frozen=True)
class VisualizationResult:
    destination: Path
    hull_vtk: Path | None
    plane_vtk: Path | None
    case_marker: Path | None
    cp_png: Path | None
    wall_shear_png: Path | None
    yplus_png: Path | None
    flow_png: Path | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisOptions:
    bow_override: str | None = None
    stop_after: Stage | None = None
    restart: bool = False
    force: bool = False
    geometry_only: bool = False
    prepare_only: bool = False
    mesh_only: MeshProfile | None = None
    reference_only: bool = False


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    status: StageStatus
    run_paths: RunPaths
    accepted_case_count: int
    diagnostic_codes: tuple[str, ...]


@dataclass(frozen=True)
class PipelineConfig:
    raw: dict[str, Any]
    digest: str
    source: Path
