from __future__ import annotations

import hashlib
import json
import math
import tomllib
from pathlib import Path
from typing import Any

from pipeline.models import PipelineConfig


def canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mapping(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"configuration section [{name}] is required")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return numeric


def _fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite fraction from 0 through 1")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be a finite fraction from 0 through 1")
    return numeric


def validate_config(raw: dict[str, Any]) -> None:
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a TOML table")
    if raw.get("schema_version") != 2:
        raise ValueError("schema_version must be 2")
    if raw.get("openfoam_version") != "2512":
        raise ValueError("OpenFOAM version must be exactly '2512'")

    mpi_ranks = raw.get("mpi_ranks")
    if isinstance(mpi_ranks, bool) or not isinstance(mpi_ranks, int) or not 1 <= mpi_ranks <= 8:
        raise ValueError("mpi_ranks must be an integer from 1 through 8")

    fluid = _mapping(raw, "fluid")
    for key, value in fluid.items():
        _positive_number(value, f"fluid.{key}")

    operating = _mapping(raw, "operating")
    speeds = operating.get("speeds_m_s")
    if not isinstance(speeds, list) or not speeds:
        raise ValueError("operating.speeds_m_s must be a non-empty list")
    normalized_speeds = [_positive_number(speed, "operating.speeds_m_s") for speed in speeds]
    if normalized_speeds != sorted(normalized_speeds) or len(set(normalized_speeds)) != len(normalized_speeds):
        raise ValueError("operating.speeds_m_s must be sorted, unique, and positive")
    reference_speed = _positive_number(operating.get("reference_speed_m_s"), "operating.reference_speed_m_s")
    if reference_speed not in normalized_speeds:
        raise ValueError("operating.reference_speed_m_s must appear in operating.speeds_m_s")

    geometry = _mapping(raw, "geometry")
    weights = [
        _positive_number(geometry.get("cap_weight"), "geometry.cap_weight"),
        _positive_number(geometry.get("taper_weight"), "geometry.taper_weight"),
        _positive_number(geometry.get("curvature_weight"), "geometry.curvature_weight"),
    ]
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("geometry cap, taper, and curvature weights must sum to 1.0")

    mesh = _mapping(raw, "mesh")
    if _positive_number(mesh.get("refinement_ratio"), "mesh.refinement_ratio") <= 1.0:
        raise ValueError("mesh.refinement_ratio must be greater than 1")
    maximum_final_layer_fraction = _fraction(
        mesh.get("maximum_final_layer_thickness_fraction"),
        "mesh.maximum_final_layer_thickness_fraction",
    )
    if maximum_final_layer_fraction == 0.0:
        raise ValueError("mesh.maximum_final_layer_thickness_fraction must be greater than 0")
    maximum_layer_growth = _positive_number(
        mesh.get("maximum_layer_growth_ratio"),
        "mesh.maximum_layer_growth_ratio",
    )
    if not 1.0 < maximum_layer_growth < 1.2:
        raise ValueError("mesh.maximum_layer_growth_ratio must be greater than 1 and less than 1.2")
    surface_layers = mesh.get("surface_layers")
    if isinstance(surface_layers, bool) or not isinstance(surface_layers, int) or surface_layers < 2:
        raise ValueError("mesh.surface_layers must be an integer of at least 2")
    wake_start = _fraction(mesh.get("wake_start_fraction"), "mesh.wake_start_fraction")
    if wake_start <= 0.0 or wake_start >= 1.0:
        raise ValueError("mesh.wake_start_fraction must be strictly between 0 and 1")

    solver = _mapping(raw, "solver")
    field_window = _positive_number(
        solver.get("field_average_window_flow_times"),
        "solver.field_average_window_flow_times",
    )
    first_checkpoint = _positive_number(
        solver.get("first_convergence_check_flow_times"),
        "solver.first_convergence_check_flow_times",
    )
    checkpoint = _positive_number(solver.get("checkpoint_flow_times"), "solver.checkpoint_flow_times")
    maximum = _positive_number(solver.get("maximum_flow_times"), "solver.maximum_flow_times")
    if field_window >= maximum:
        raise ValueError("solver field-average window must be below maximum flow-through time")
    if first_checkpoint >= maximum:
        raise ValueError("solver first convergence check must be below maximum flow-through time")
    if checkpoint > first_checkpoint:
        raise ValueError("solver checkpoint interval cannot exceed the first convergence check time")
    _fraction(
        solver.get("drag_running_mean_uncertainty_fraction"),
        "solver.drag_running_mean_uncertainty_fraction",
    )
    _fraction(solver.get("drag_mean_drift_fraction"), "solver.drag_mean_drift_fraction")
    target_courant = _positive_number(solver.get("target_courant_number"), "solver.target_courant_number")
    maximum_courant = _positive_number(solver.get("maximum_courant_number"), "solver.maximum_courant_number")
    if target_courant > maximum_courant:
        raise ValueError("solver.target_courant_number cannot exceed solver.maximum_courant_number")
    in_range = _fraction(
        solver.get("minimum_yplus_30_to_300_fraction"),
        "solver.minimum_yplus_30_to_300_fraction",
    )
    preferred_high_tail = _fraction(
        solver.get("preferred_yplus_above_300_fraction"),
        "solver.preferred_yplus_above_300_fraction",
    )
    maximum_high_tail = _fraction(
        solver.get("maximum_yplus_above_300_fraction"),
        "solver.maximum_yplus_above_300_fraction",
    )
    if preferred_high_tail > maximum_high_tail:
        raise ValueError("preferred yPlus high-tail fraction cannot exceed the blocking maximum")
    if in_range + maximum_high_tail < 1.0:
        raise ValueError("yPlus in-range and blocking high-tail fractions leave an unqualified acceptance gap")


def load_config(path: Path) -> PipelineConfig:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    validate_config(raw)
    return PipelineConfig(raw=raw, digest=canonical_digest(raw), source=path.resolve())
