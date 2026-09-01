from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from pipeline.models import PipelineConfig


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_VECTOR = re.compile(rf"\(\s*({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*\)")
_EXPECTED_RESIDUAL_FIELDS = ("p", "Ux", "Uy", "Uz", "k", "omega")
_ABSOLUTE_INITIAL_RESIDUAL_LIMITS = {
    "p": 1.0e-4,
    "Ux": 2.0e-4,
    "Uy": 2.0e-4,
    "Uz": 2.0e-4,
    "k": 1.0e-5,
    "omega": 1.0e-5,
}


def absolute_initial_residual_limit(field: str) -> float:
    try:
        return _ABSOLUTE_INITIAL_RESIDUAL_LIMITS[field]
    except KeyError as error:
        raise ValueError(f"no absolute residual limit is defined for {field}") from error


@dataclass(frozen=True)
class ForceHistory:
    times_s: np.ndarray
    total_force_n: np.ndarray
    pressure_force_n: np.ndarray
    viscous_force_n: np.ndarray
    total_moment_nm: np.ndarray


@dataclass(frozen=True)
class ResidualHistory:
    times_s: np.ndarray
    initial: np.ndarray
    final: np.ndarray


@dataclass(frozen=True)
class ConvergenceStatistics:
    start_index: int
    end_index: int
    mean_drag_n: float
    drag_cov: float
    drag_drift: float
    mean_uncertainty_n: float
    mean_uncertainty_fraction: float
    integral_time_scale_s: float
    effective_sample_count: float
    duration_s: float
    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class WallSurfaceData:
    points: np.ndarray
    cells: tuple[np.ndarray, ...]
    point_fields: dict[str, np.ndarray]
    cell_fields: dict[str, np.ndarray]


def _segment_key(path: Path) -> tuple[float, str]:
    try:
        start = float(path.parent.name)
    except ValueError:
        start = -math.inf
    return start, path.as_posix()


def _files(path: Path, name: str) -> list[Path]:
    source = Path(path).resolve(strict=True)
    if source.is_file():
        return [source] if source.name == name else []
    return sorted((candidate for candidate in source.rglob(name) if candidate.is_file()), key=_segment_key)


def _vectors(line: str) -> list[np.ndarray]:
    return [np.asarray(tuple(float(value) for value in match), dtype=float) for match in _VECTOR.findall(line)]


def _force_row_vectors(line: str, path: Path, iteration: float) -> list[np.ndarray]:
    """Read both parenthesized and v2512 flat force/moment rows."""
    vectors = _vectors(line)
    if vectors:
        return vectors
    tokens = line.split()
    try:
        scalars = [float(token) for token in tokens]
    except ValueError as error:
        raise ValueError(f"unsupported OpenFOAM force row at time {iteration:g}: {path}") from error
    if len(scalars) not in (9, 12, 18):
        raise ValueError(f"unsupported OpenFOAM force row at time {iteration:g}: {path}")
    return [np.asarray(scalars[index : index + 3], dtype=float) for index in range(0, len(scalars), 3)]


def _parse_force_files(paths: Iterable[Path]) -> tuple[
    dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    dict[float, np.ndarray],
]:
    forces: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    embedded_moments: dict[float, np.ndarray] = {}
    for path in paths:
        legacy = path.name == "forces.dat"
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            time_match = re.match(rf"^({_FLOAT})(?:\s|$)", line)
            if time_match is None:
                continue
            iteration = float(time_match.group(1))
            values = _force_row_vectors(line[time_match.end() :], path, iteration)
            if legacy and len(values) >= 6:
                pressure, viscous, porous = values[:3]
                moment_pressure, moment_viscous, moment_porous = values[3:6]
                if not np.allclose(porous, 0.0, rtol=0.0, atol=1.0e-12):
                    raise ValueError(f"non-zero porous force is outside this hull model: {path}")
                if not np.allclose(moment_porous, 0.0, rtol=0.0, atol=1.0e-12):
                    raise ValueError(f"non-zero porous moment is outside this hull model: {path}")
                forces[iteration] = (pressure + viscous, pressure, viscous)
                embedded_moments[iteration] = moment_pressure + moment_viscous
                continue
            if len(values) not in (3, 4):
                raise ValueError(f"unsupported OpenFOAM force row at time {iteration:g}: {path}")
            total, pressure, viscous = values[:3]
            porous = np.zeros(3, dtype=float) if len(values) == 3 else values[3]
            if not np.allclose(porous, 0.0, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"non-zero porous force is outside this hull model: {path}")
            if not np.allclose(total, pressure + viscous, rtol=2.0e-5, atol=1.0e-7):
                raise ValueError(f"total force does not equal pressure plus viscous force at time {iteration:g}: {path}")
            forces[iteration] = (total, pressure, viscous)
    return forces, embedded_moments


def _parse_moment_files(paths: Iterable[Path]) -> dict[float, np.ndarray]:
    moments: dict[float, np.ndarray] = {}
    for path in paths:
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            time_match = re.match(rf"^({_FLOAT})(?:\s|$)", line)
            if time_match is None:
                continue
            iteration = float(time_match.group(1))
            values = _force_row_vectors(line[time_match.end() :], path, iteration)
            if len(values) not in (3, 4):
                raise ValueError(f"unsupported OpenFOAM moment row at time {iteration:g}: {path}")
            total, pressure, viscous = values[:3]
            porous = np.zeros(3, dtype=float) if len(values) == 3 else values[3]
            if not np.allclose(porous, 0.0, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"non-zero porous moment is outside this hull model: {path}")
            if not np.allclose(total, pressure + viscous, rtol=2.0e-5, atol=1.0e-7):
                raise ValueError(f"total moment does not equal pressure plus viscous moment at time {iteration:g}: {path}")
            moments[iteration] = total
    return moments


def parse_force_history(path: Path) -> ForceHistory:
    root = Path(path).resolve(strict=True)
    legacy_files = _files(root, "forces.dat")
    force_files = legacy_files or _files(root, "force.dat")
    if not force_files:
        raise ValueError(f"no OpenFOAM force history found under {root}")
    forces, moments = _parse_force_files(force_files)
    if not legacy_files:
        moments.update(_parse_moment_files(_files(root, "moment.dat")))
    common = sorted(set(forces).intersection(moments))
    if not common:
        raise ValueError(f"force and moment histories have no common samples under {root}")
    total = np.asarray([forces[time][0] for time in common], dtype=float)
    pressure = np.asarray([forces[time][1] for time in common], dtype=float)
    viscous = np.asarray([forces[time][2] for time in common], dtype=float)
    moment = np.asarray([moments[time] for time in common], dtype=float)
    if not all(np.all(np.isfinite(values)) for values in (total, pressure, viscous, moment)):
        raise ValueError(f"force history contains non-finite values under {root}")
    return ForceHistory(
        times_s=np.asarray(common, dtype=float),
        total_force_n=total,
        pressure_force_n=pressure,
        viscous_force_n=viscous,
        total_moment_nm=moment,
    )


def parse_solver_info(path: Path) -> dict[str, ResidualHistory]:
    root = Path(path).resolve(strict=True)
    files = _files(root, "solverInfo.dat")
    if not files:
        raise ValueError(f"no OpenFOAM solverInfo.dat found under {root}")
    samples: dict[str, dict[float, tuple[float, float]]] = {}
    for info_path in files:
        header: list[str] | None = None
        for raw_line in info_path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("#"):
                candidate = stripped.lstrip("# ").split()
                if candidate and candidate[0] == "Time":
                    header = candidate
                continue
            if not stripped or header is None:
                continue
            columns = stripped.split()
            if len(columns) != len(header):
                raise ValueError(f"solverInfo row/header column mismatch: {info_path}")
            iteration = float(columns[0])
            for index, name in enumerate(header):
                if not name.endswith("_initial"):
                    continue
                field = name.removesuffix("_initial")
                final_name = f"{field}_final"
                try:
                    final_index = header.index(final_name)
                except ValueError as error:
                    raise ValueError(f"solverInfo lacks {final_name}: {info_path}") from error
                initial = float(columns[index])
                final = float(columns[final_index])
                if not math.isfinite(initial) or not math.isfinite(final) or initial < 0.0 or final < 0.0:
                    raise ValueError(f"solverInfo has invalid residuals for {field} at {iteration:g}: {info_path}")
                samples.setdefault(field, {})[iteration] = (initial, final)
    missing = [field for field in _EXPECTED_RESIDUAL_FIELDS if field not in samples]
    if missing:
        raise ValueError(f"solverInfo lacks required fields: {', '.join(missing)}")
    histories: dict[str, ResidualHistory] = {}
    for field in _EXPECTED_RESIDUAL_FIELDS:
        times = sorted(samples[field])
        histories[field] = ResidualHistory(
            times_s=np.asarray(times, dtype=float),
            initial=np.asarray([samples[field][time][0] for time in times], dtype=float),
            final=np.asarray([samples[field][time][1] for time in times], dtype=float),
        )
    return histories


def _reduction_orders(first: float, last: float) -> float:
    if first == 0.0:
        return -math.log10(np.finfo(float).tiny) if last == 0.0 else -math.log10(last / np.finfo(float).tiny)
    return math.log10(first / max(last, np.finfo(float).tiny))


def _uniform_series(times_s: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(times_s) < 3 or len(times_s) != len(values):
        raise ValueError("statistical convergence requires at least three paired force samples")
    increments = np.diff(times_s)
    if not np.all(np.isfinite(increments)) or np.any(increments <= 0.0):
        raise ValueError("force sample times must be finite and strictly increasing")
    spacing = float(np.median(increments))
    count = max(3, int(round((float(times_s[-1]) - float(times_s[0])) / spacing)) + 1)
    uniform_times = np.linspace(float(times_s[0]), float(times_s[-1]), count)
    return uniform_times, np.interp(uniform_times, times_s, values)


def _correlated_mean_statistics(times_s: np.ndarray, values: np.ndarray) -> tuple[float, float, float, float]:
    uniform_times, uniform_values = _uniform_series(times_s, values)
    count = len(uniform_values)
    mean = float(np.mean(uniform_values))
    standard_deviation = float(np.std(uniform_values, ddof=1))
    spacing = float(uniform_times[1] - uniform_times[0])
    if standard_deviation <= float(np.finfo(float).eps) * max(abs(mean), 1.0):
        return 0.0, float(count), 0.5 * spacing, standard_deviation
    centred = uniform_values - mean
    correlation = np.correlate(centred, centred, mode="full")[count - 1 :]
    correlation /= correlation[0]
    positive = correlation[1:]
    non_positive = np.flatnonzero(positive <= 0.0)
    stop = int(non_positive[0]) if non_positive.size else len(positive)
    correlation_sum = float(np.sum(positive[:stop]))
    variance_inflation = max(1.0, 1.0 + 2.0 * correlation_sum)
    effective_samples = min(float(count), float(count) / variance_inflation)
    standard_error = standard_deviation / math.sqrt(effective_samples)
    integral_time = 0.5 * variance_inflation * spacing
    return standard_error, effective_samples, integral_time, standard_deviation


def _automatic_transient_index(values: np.ndarray) -> int:
    """Locate the initial transient using the minimum-uncertainty criterion."""
    count = len(values)
    if count < 3:
        raise ValueError("transient detection requires at least three force samples")
    suffix_sum = np.cumsum(values[::-1], dtype=float)[::-1]
    suffix_square_sum = np.cumsum(np.square(values[::-1]), dtype=float)[::-1]
    candidates = np.arange(count // 2 + 1, dtype=int)
    remaining = count - candidates
    squared_deviation = suffix_square_sum[candidates] - np.square(suffix_sum[candidates]) / remaining
    objective = np.maximum(squared_deviation, 0.0) / np.square(np.maximum(remaining - 1, 1))
    return int(candidates[int(np.argmin(objective))])


def select_converged_statistics(
    force_history: ForceHistory,
    config: PipelineConfig,
    flow_time_s: float,
    maximum_time_s: float | None = None,
) -> ConvergenceStatistics:
    policy = config.raw["solver"]
    if not math.isfinite(flow_time_s) or flow_time_s <= 0.0:
        raise ValueError("flow_time_s must be finite and positive")
    upper = float(force_history.times_s[-1]) if maximum_time_s is None else float(maximum_time_s)
    selected = np.flatnonzero(force_history.times_s <= upper)
    reasons: list[str] = []
    first_checkpoint = float(policy["first_convergence_check_flow_times"]) * flow_time_s
    if upper < first_checkpoint - 1.0e-9 * flow_time_s:
        reasons.append("insufficient_flow_through_time")
    if selected.size < 3:
        return ConvergenceStatistics(
            0, 0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, 0.0,
            False, tuple((*reasons, "insufficient_force_samples")),
        )
    available_times = force_history.times_s[selected]
    available_drag = force_history.total_force_n[selected, 0]
    uniform_available_times, uniform_available_drag = _uniform_series(available_times, available_drag)
    transient_index = _automatic_transient_index(uniform_available_drag)
    transient_end_time = float(uniform_available_times[transient_index])
    start_index = int(selected[np.searchsorted(available_times, transient_end_time, side="left")])
    end_index = int(selected[-1]) + 1
    times = force_history.times_s[start_index:end_index]
    drag = force_history.total_force_n[start_index:end_index, 0]
    duration = float(times[-1] - times[0])
    uniform_times, uniform_drag = _uniform_series(times, drag)
    mean = float(np.mean(uniform_drag))
    if not math.isfinite(mean) or mean <= 0.0:
        reasons.append("non_positive_or_non_finite_drag")
    denominator = abs(mean)
    covariance = math.inf if denominator == 0.0 else float(np.std(uniform_drag, ddof=1) / denominator)
    midpoint = len(uniform_drag) // 2
    first_mean = float(np.mean(uniform_drag[:midpoint]))
    second_mean = float(np.mean(uniform_drag[midpoint:]))
    drift = math.inf if denominator == 0.0 else abs(second_mean - first_mean) / denominator
    running_mean = np.cumsum(uniform_drag, dtype=float) / np.arange(1, len(uniform_drag) + 1, dtype=float)
    settled_running_mean = running_mean[len(running_mean) // 2 :]
    mean_uncertainty = 0.5 * float(np.max(settled_running_mean) - np.min(settled_running_mean))
    mean_uncertainty_fraction = math.inf if denominator == 0.0 else mean_uncertainty / denominator
    _, effective_samples, integral_time, _ = _correlated_mean_statistics(uniform_times, uniform_drag)
    if mean_uncertainty_fraction > float(policy["drag_running_mean_uncertainty_fraction"]):
        reasons.append("drag_running_mean_uncertainty")
    if drift > float(policy["drag_mean_drift_fraction"]):
        reasons.append("drag_mean_drift")
    return ConvergenceStatistics(
        start_index=start_index,
        end_index=end_index,
        mean_drag_n=mean,
        drag_cov=covariance,
        drag_drift=drift,
        mean_uncertainty_n=mean_uncertainty,
        mean_uncertainty_fraction=mean_uncertainty_fraction,
        integral_time_scale_s=integral_time,
        effective_sample_count=effective_samples,
        duration_s=duration,
        accepted=not reasons,
        reasons=tuple(reasons),
    )


def summarize_residuals(
    residuals: dict[str, ResidualHistory],
    config: PipelineConfig,
    window_start_time_s: float,
    maximum_time_s: float,
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for field in _EXPECTED_RESIDUAL_FIELDS:
        history = residuals[field]
        indices = np.flatnonzero((history.times_s > window_start_time_s) & (history.times_s <= maximum_time_s))
        if indices.size < 2:
            raise ValueError(f"insufficient accepted-window residual samples for {field}")
        initial = history.initial[indices]
        final = history.final[indices]
        summary[field] = {
            "first_initial": float(initial[0]),
            "last_initial": float(initial[-1]),
            "minimum_initial": float(np.min(initial)),
            "maximum_initial": float(np.max(initial)),
            "first_final": float(final[0]),
            "last_final": float(final[-1]),
            "minimum_final": float(np.min(final)),
            "maximum_final": float(np.max(final)),
            "time_series_change_orders": _reduction_orders(float(initial[0]), float(initial[-1])),
            "absolute_initial_limit": absolute_initial_residual_limit(field),
            "absolute_initial_passed": float(float(initial[-1]) <= absolute_initial_residual_limit(field)),
        }
    return summary


def _cell_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    origin = points[0]
    return float(
        sum(
            0.5 * np.linalg.norm(np.cross(points[index] - origin, points[index + 1] - origin))
            for index in range(1, len(points) - 1)
        )
    )


def _matching_field(fields: Iterable[str], requested: str) -> str:
    candidates = [name for name in fields if name == requested or name.rsplit(":", 1)[-1] == requested]
    if len(candidates) != 1:
        raise ValueError(f"expected one VTK field named {requested!r}, found {candidates}")
    return candidates[0]


def _field_data(
    tokens: list[str], index: int, tuple_count: int
) -> tuple[dict[str, np.ndarray], int]:
    if tokens[index] != "FIELD" or tokens[index + 1] != "FieldData":
        raise ValueError("associated FIELD section is missing")
    field_count = int(tokens[index + 2])
    index += 3
    fields: dict[str, np.ndarray] = {}
    for _ in range(field_count):
        name = tokens[index]
        component_count = int(tokens[index + 1])
        actual_tuple_count = int(tokens[index + 2])
        index += 4  # field name, components, tuples, numeric type
        value_count = component_count * actual_tuple_count
        values = np.asarray([float(value) for value in tokens[index : index + value_count]], dtype=float)
        index += value_count
        if actual_tuple_count != tuple_count:
            raise ValueError(f"VTK field {name!r} tuple count is inconsistent")
        fields[name] = values.reshape((actual_tuple_count, component_count))
    return fields, index


def read_wall_surface(path: Path) -> WallSurfaceData:
    """Read ASCII legacy POLYDATA emitted by OpenFOAM surface writers."""
    vtk_path = Path(path).resolve(strict=True)
    text = vtk_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 4 or lines[2].strip() != "ASCII" or lines[3].strip() != "DATASET POLYDATA":
        raise ValueError(f"expected ASCII VTK POLYDATA hull surface: {vtk_path}")
    tokens = text.split()
    try:
        index = tokens.index("POINTS")
        point_count = int(tokens[index + 1])
        index += 3
        point_end = index + 3 * point_count
        points = np.asarray([float(value) for value in tokens[index:point_end]], dtype=float).reshape((-1, 3))
        index = point_end
        if tokens[index] != "POLYGONS":
            raise ValueError("POLYGONS section is missing")
        cell_count = int(tokens[index + 1])
        connectivity_count = int(tokens[index + 2])
        index += 3
        connectivity_end = index + connectivity_count
        cells: list[np.ndarray] = []
        while index < connectivity_end:
            vertex_count = int(tokens[index])
            index += 1
            cells.append(np.asarray([int(value) for value in tokens[index : index + vertex_count]], dtype=int))
            index += vertex_count
        if index != connectivity_end or len(cells) != cell_count:
            raise ValueError("POLYGONS connectivity count is inconsistent")
        point_fields: dict[str, np.ndarray] = {}
        cell_fields: dict[str, np.ndarray] = {}
        while index < len(tokens):
            association = tokens[index]
            if association not in {"POINT_DATA", "CELL_DATA"}:
                raise ValueError(f"unsupported VTK data association {association!r}")
            tuple_count = int(tokens[index + 1])
            index += 2
            fields, index = _field_data(tokens, index, tuple_count)
            if association == "POINT_DATA":
                if tuple_count != point_count:
                    raise ValueError("POINT_DATA count is inconsistent")
                point_fields.update(fields)
            else:
                if tuple_count != cell_count:
                    raise ValueError("CELL_DATA count is inconsistent")
                cell_fields.update(fields)
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid ASCII VTK POLYDATA hull surface: {vtk_path}: {error}") from error
    all_fields = (*point_fields.values(), *cell_fields.values())
    if not np.all(np.isfinite(points)) or any(not np.all(np.isfinite(values)) for values in all_fields):
        raise ValueError(f"VTK hull surface contains non-finite data: {vtk_path}")
    return WallSurfaceData(
        points=points,
        cells=tuple(cells),
        point_fields=point_fields,
        cell_fields=cell_fields,
    )


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    threshold = percentile * float(np.sum(ordered_weights))
    index = int(np.searchsorted(np.cumsum(ordered_weights), threshold, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def wall_field_statistics(path: Path) -> tuple[dict[str, float], float]:
    vtk_path = Path(path).resolve(strict=True)
    surface = read_wall_surface(vtk_path)
    cell_fields = surface.cell_fields
    yplus_name = _matching_field(cell_fields, "yPlus")
    cp_name = _matching_field(cell_fields, "cpMean")
    shear_name = _matching_field(cell_fields, "wallShearStressMean")
    areas: list[float] = []
    yplus_values: list[float] = []
    cp_values: list[float] = []
    yplus = np.asarray(cell_fields[yplus_name], dtype=float).reshape(-1)
    cp = np.asarray(cell_fields[cp_name], dtype=float).reshape(-1)
    shear = np.asarray(cell_fields[shear_name], dtype=float)
    if len(surface.cells) != len(yplus) or len(surface.cells) != len(cp) or len(surface.cells) != len(shear):
        raise ValueError(f"VTK field/cell count mismatch: {vtk_path}")
    for cell, yplus_value, cp_value in zip(surface.cells, yplus, cp, strict=True):
        area = _cell_area(np.asarray(surface.points[np.asarray(cell, dtype=int)], dtype=float))
        if area <= 0.0 or not math.isfinite(area):
            raise ValueError(f"VTK hull contains a non-positive cell area: {vtk_path}")
        areas.append(area)
        yplus_values.append(float(yplus_value))
        cp_values.append(float(cp_value))
    weights = np.asarray(areas, dtype=float)
    yplus = np.asarray(yplus_values, dtype=float)
    cp = np.asarray(cp_values, dtype=float)
    if yplus.size == 0 or not np.all(np.isfinite(yplus)) or not np.all(np.isfinite(cp)):
        raise ValueError(f"VTK wall fields are empty or non-finite: {vtk_path}")
    total_area = float(np.sum(weights))
    stats = {
        "minimum": float(np.min(yplus)),
        "p05": _weighted_percentile(yplus, weights, 0.05),
        "median": _weighted_percentile(yplus, weights, 0.50),
        "mean": float(np.average(yplus, weights=weights)),
        "p95": _weighted_percentile(yplus, weights, 0.95),
        "maximum": float(np.max(yplus)),
        "fraction_below_30": float(np.sum(weights[yplus < 30.0]) / total_area),
        "fraction_30_to_300": float(np.sum(weights[(yplus >= 30.0) & (yplus <= 300.0)]) / total_area),
        "fraction_above_300": float(np.sum(weights[yplus > 300.0]) / total_area),
        "area_m2": total_area,
    }
    return stats, float(np.min(cp))
