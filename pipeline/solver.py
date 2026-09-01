from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from pipeline.models import CaseResult, GeometryResult, MeshProfile, MeshResult, OpenFoamRuntime, PipelineConfig
from pipeline.postprocess import (
    parse_force_history,
    parse_solver_info,
    select_converged_statistics,
    summarize_residuals,
    wall_field_statistics,
)
from pipeline.runtime import CommandFailure, require_executable, run_checked
from pipeline.state import atomic_write_json, sha256_file


_NON_FINITE = re.compile(r"(?i)(?<![A-Za-z])(?:nan|inf|infinity)(?![A-Za-z])")
_COURANT = re.compile(
    r"Courant Number mean:\s*[-+0-9.eE]+\s+max:\s*([-+0-9.eE]+)",
    re.IGNORECASE,
)
_SOLVER_FAILURE_PATTERNS = (
    (
        "floating_point_exception",
        re.compile(r"(?:floating point exception(?! trapping enabled)|Foam::sigFpe::sigHandler)", re.IGNORECASE),
    ),
    ("non_finite_solver_value", _NON_FINITE),
    ("continuity_blow_up", re.compile(r"continuity[^\n]*(?:blow|diverg|cannot be removed)", re.IGNORECASE)),
)


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, MeshProfile):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _link_mesh(case_dir: Path, mesh: MeshResult) -> Path:
    source_root = mesh.case_dir.resolve(strict=True) / "constant" / "polyMesh"
    if not source_root.is_dir():
        raise ValueError(f"accepted source polyMesh is unavailable: {source_root}")
    destination_root = case_dir / "constant" / "polyMesh"
    if destination_root.exists():
        raise ValueError(f"solver polyMesh destination already exists: {destination_root}")
    destination_root.mkdir(parents=True)
    entries: list[dict[str, object]] = []
    same_filesystem = source_root.stat().st_dev == destination_root.stat().st_dev
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if same_filesystem:
            os.link(source, destination)
            method = "hardlink"
        else:
            shutil.copy2(source, destination)
            method = "copy"
        digest = sha256_file(source)
        if sha256_file(destination) != digest:
            raise ValueError(f"solver mesh file digest mismatch: {destination}")
        entries.append(
            {
                "relative_path": relative.as_posix(),
                "source": str(source),
                "destination": str(destination),
                "sha256": digest,
                "method": method,
            }
        )
    if not entries:
        raise ValueError(f"accepted source polyMesh contains no files: {source_root}")
    manifest = case_dir / "mesh-source.json"
    atomic_write_json(
        manifest,
        {
            "schema_version": 2,
            "source_mesh_sha256": mesh.mesh_sha256,
            "source_profile": mesh.profile.value,
            "source_case": str(mesh.case_dir),
            "same_filesystem": same_filesystem,
            "files": entries,
        },
    )
    return manifest


def _run(argv: Sequence[str | Path], case_dir: Path, log_name: str) -> Path:
    log = case_dir / log_name
    run_checked(argv, case_dir, log)
    return log


def _parallel_argv(runtime: OpenFoamRuntime, ranks: int, executable: Path, *arguments: str | Path) -> list[str | Path]:
    return [
        runtime.mpirun,
        "--bind-to",
        "core",
        "--map-by",
        "core",
        "-np",
        str(ranks),
        executable,
        *arguments,
    ]


def _processor_latest_time(case_dir: Path, ranks: int) -> float:
    latest: list[float] = []
    for rank in range(ranks):
        processor = case_dir / f"processor{rank}"
        if not processor.is_dir():
            raise ValueError(f"parallel solver directory is missing: {processor}")
        times: list[float] = []
        for candidate in processor.iterdir():
            if not candidate.is_dir():
                continue
            try:
                times.append(float(candidate.name))
            except ValueError:
                continue
        if not times:
            raise ValueError(f"parallel solver produced no numeric time under {processor}")
        latest.append(max(times))
    tolerance = max(1.0e-10, 1.0e-10 * max(latest))
    if max(latest) - min(latest) > tolerance:
        raise ValueError(f"parallel ranks ended at different physical times: {latest}")
    return latest[0]


def _write_control_end_time(case_dir: Path, end_time_s: float) -> None:
    control = case_dir / "system" / "controlDict"
    text = control.read_text(encoding="utf-8")
    text, start_count = re.subn(r"(?m)^startFrom\s+\w+\s*;", "startFrom latestTime;", text)
    text, end_count = re.subn(
        r"(?m)^endTime\s+[-+0-9.eE]+\s*;",
        f"endTime {end_time_s:.12g};",
        text,
    )
    if start_count != 1 or end_count != 1:
        raise ValueError(
            "transient continuation controlDict replacement matched "
            f"startFrom={start_count}, endTime={end_count} entries"
        )
    descriptor, temporary_name = tempfile.mkstemp(prefix=".controlDict.", suffix=".tmp", dir=control.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, control)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _solver_diagnostics(logs: Sequence[Path]) -> tuple[str, ...]:
    diagnostics: list[str] = []
    for log in logs:
        text = log.read_text(encoding="utf-8", errors="replace")
        for code, pattern in _SOLVER_FAILURE_PATTERNS:
            if pattern.search(text) and code not in diagnostics:
                diagnostics.append(code)
    return tuple(diagnostics)


def _maximum_courant_number(logs: Sequence[Path]) -> float:
    values: list[float] = []
    for log in logs:
        text = log.read_text(encoding="utf-8", errors="replace")
        marker = "Starting time loop"
        if marker not in text:
            raise ValueError(f"pimpleFoam log lacks transient-loop marker: {log}")
        transient_text = text.split(marker, maxsplit=1)[1]
        values.extend(float(value) for value in _COURANT.findall(transient_text))
    if not values or not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("pimpleFoam logs contain no valid Courant-number history")
    return max(values)


def _time_policy(
    speed_m_s: float,
    geometry: GeometryResult,
    config: PipelineConfig,
) -> dict[str, float]:
    if not math.isfinite(speed_m_s) or speed_m_s <= 0.0:
        raise ValueError("speed_m_s must be finite and positive")
    flow_time = geometry.length_m / speed_m_s
    policy = config.raw["solver"]
    return {
        "flow_time_s": flow_time,
        "time_step_s": float(policy["time_step_flow_fraction"]) * flow_time,
        "field_average_window_time_s": float(policy["field_average_window_flow_times"]) * flow_time,
        "first_checkpoint_time_s": float(policy["first_convergence_check_flow_times"]) * flow_time,
        "checkpoint_time_s": float(policy["checkpoint_flow_times"]) * flow_time,
        "maximum_time_s": float(policy["maximum_flow_times"]) * flow_time,
    }


def _checkpoint_targets(time_policy: dict[str, float]) -> tuple[float, ...]:
    first = time_policy["first_checkpoint_time_s"]
    step = time_policy["checkpoint_time_s"]
    maximum = time_policy["maximum_time_s"]
    targets: list[float] = []
    current = first
    tolerance = 1.0e-10 * max(maximum, 1.0)
    while current < maximum - tolerance:
        targets.append(current)
        current += step
    targets.append(maximum)
    return tuple(targets)


def _write_convergence_checkpoint(
    case_dir: Path,
    target_time_s: float,
    flow_time_s: float,
    config: PipelineConfig,
    solver_logs: Sequence[Path],
) -> tuple[bool, bool]:
    destination = case_dir / "convergence-checkpoints" / f"{target_time_s:.12g}.json"
    failure_codes = list(_solver_diagnostics(solver_logs))
    try:
        force_history = parse_force_history(case_dir / "postProcessing" / "forces")
        statistics = select_converged_statistics(
            force_history,
            config,
            flow_time_s=flow_time_s,
            maximum_time_s=target_time_s,
        )
        maximum_courant = _maximum_courant_number(solver_logs)
        courant_limit = float(config.raw["solver"]["maximum_courant_number"])
        if maximum_courant > courant_limit:
            failure_codes.append("courant_number_limit")
        accepted = statistics.accepted and not failure_codes
        atomic_write_json(
            destination,
            {
                "schema_version": 3,
                "physical_time_s": target_time_s,
                "flow_through_times": target_time_s / flow_time_s,
                "accepted": accepted,
                "solver_failure_codes": list(dict.fromkeys(failure_codes)),
                "maximum_courant_number": maximum_courant,
                "maximum_courant_number_limit": courant_limit,
                "stationary_sample": {
                    "start_time_s": (
                        None
                        if statistics.end_index == 0
                        else float(force_history.times_s[statistics.start_index])
                    ),
                    "end_time_s": (
                        None
                        if statistics.end_index == 0
                        else float(force_history.times_s[statistics.end_index - 1])
                    ),
                    "duration_s": statistics.duration_s,
                    "duration_flow_through_times": statistics.duration_s / flow_time_s,
                    "mean_drag_n": statistics.mean_drag_n,
                    "coefficient_of_variation": statistics.drag_cov,
                    "mean_drift_fraction": statistics.drag_drift,
                    "running_mean_uncertainty_n": statistics.mean_uncertainty_n,
                    "running_mean_uncertainty_fraction": statistics.mean_uncertainty_fraction,
                    "integral_time_scale_diagnostic_s": statistics.integral_time_scale_s,
                    "effective_sample_count_diagnostic": statistics.effective_sample_count,
                    "accepted": statistics.accepted,
                    "reasons": list(statistics.reasons),
                },
                "decision": "stop_statistically_converged" if accepted else "continue_or_reject",
            },
        )
        return accepted, bool(failure_codes)
    except ValueError as error:
        atomic_write_json(
            destination,
            {
                "schema_version": 3,
                "physical_time_s": target_time_s,
                "flow_through_times": target_time_s / flow_time_s,
                "accepted": False,
                "solver_failure_codes": list(dict.fromkeys(failure_codes)),
                "decision": "continue_or_reject",
                "evaluation_error": str(error),
            },
        )
        return False, bool(failure_codes)


def export_wall_fields(case_dir: Path, runtime: OpenFoamRuntime) -> Path:
    function_log = _run(
        [
            runtime.pimple_foam,
            "-case",
            case_dir,
            "-postProcess",
            "-latestTime",
            "-func",
            "yPlus",
        ],
        case_dir,
        "log.yPlus.postProcess",
    )
    vtk_root = case_dir / "VTK-wall-ascii"
    _run(
        [
            require_executable("foamToVTK"),
            "-case",
            case_dir,
            "-latestTime",
            "-no-internal",
            "-patches",
            "(hull)",
            "-fields",
            "(yPlus cpMean wallShearStressMean)",
            "-legacy",
            "-ascii",
            "-no-point-data",
            "-noFunctionObjects",
            "-name",
            vtk_root.name,
            "-overwrite",
        ],
        case_dir,
        "log.foamToVTK.wall",
    )
    candidates = sorted(
        (
            path
            for path in vtk_root.rglob("*")
            if path.suffix.lower() in (".vtp", ".vtk")
            and path.is_file()
            and (path.stem == "hull" or path.stem.startswith("hull_") or "hull" in path.parts)
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not candidates:
        raise ValueError(f"foamToVTK did not export averaged hull fields under {vtk_root}; log: {function_log}")
    return candidates[-1]


def _hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths if path.is_file()}


def _time_average(times_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(times_s) < 2 or len(times_s) != len(values):
        raise ValueError("time average requires at least two paired samples")
    duration = float(times_s[-1] - times_s[0])
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("time-average duration must be finite and positive")
    return np.asarray(np.trapezoid(values, x=times_s, axis=0) / duration, dtype=float)


def case_result_payload(result: CaseResult) -> dict[str, object]:
    return _jsonable(asdict(result))  # type: ignore[return-value]


def case_result_from_json(path: Path) -> CaseResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 3:
        raise ValueError(f"case result is not from the current transient protocol: {path}")
    value = payload.get("result")
    if not isinstance(value, dict):
        raise ValueError(f"case result has invalid structure: {path}")
    return CaseResult(
        case_dir=Path(str(value["case_dir"])).resolve(strict=True),
        speed_m_s=float(value["speed_m_s"]),
        mesh_profile=MeshProfile(str(value["mesh_profile"])),
        mesh_sha256=str(value["mesh_sha256"]),
        reynolds_number=float(value["reynolds_number"]),
        accepted=bool(value["accepted"]),
        rejection_code=None if value.get("rejection_code") is None else str(value["rejection_code"]),
        diagnostic_codes=tuple(str(item) for item in value["diagnostic_codes"]),
        force_total_n=tuple(float(item) for item in value["force_total_n"]),
        force_pressure_n=tuple(float(item) for item in value["force_pressure_n"]),
        force_viscous_n=tuple(float(item) for item in value["force_viscous_n"]),
        moment_nm=tuple(float(item) for item in value["moment_nm"]),
        drag_n=float(value["drag_n"]),
        drag_coefficient=float(value["drag_coefficient"]),
        tow_power_w=float(value["tow_power_w"]),
        force_stationary_mean_n=float(value["force_stationary_mean_n"]),
        force_stationary_cov=float(value["force_stationary_cov"]),
        force_mean_drift=float(value["force_mean_drift"]),
        force_stationary_start_time_s=float(value["force_stationary_start_time_s"]),
        force_stationary_end_time_s=float(value["force_stationary_end_time_s"]),
        force_stationary_duration_flow_times=float(value["force_stationary_duration_flow_times"]),
        force_mean_uncertainty_n=float(value["force_mean_uncertainty_n"]),
        force_mean_uncertainty_fraction=float(value["force_mean_uncertainty_fraction"]),
        force_integral_time_scale_s=float(value["force_integral_time_scale_s"]),
        force_effective_sample_count=float(value["force_effective_sample_count"]),
        yplus={str(key): float(item) for key, item in value["yplus"].items()},
        cp_min=float(value["cp_min"]),
        minimum_absolute_pressure_pa=float(value["minimum_absolute_pressure_pa"]),
        cavitation_margin_pa=float(value["cavitation_margin_pa"]),
        residuals={
            str(field): {str(key): float(item) for key, item in statistics.items()}
            for field, statistics in value["residuals"].items()
        },
        physical_time_s=float(value["physical_time_s"]),
        time_steps=int(value["time_steps"]),
        maximum_courant_number=float(value["maximum_courant_number"]),
        force_history_path=Path(str(value["force_history_path"])).resolve(strict=True),
        solver_info_path=Path(str(value["solver_info_path"])).resolve(strict=True),
        wall_fields_path=Path(str(value["wall_fields_path"])).resolve(strict=True),
    )


def postprocess_completed_case(
    case_dir: Path,
    speed_m_s: float,
    mesh: MeshResult,
    geometry: GeometryResult,
    runtime: OpenFoamRuntime,
    config: PipelineConfig,
) -> CaseResult:
    validated_case = Path(case_dir).resolve(strict=True)
    ranks = int(config.raw["mpi_ranks"])
    final_time = _processor_latest_time(validated_case, ranks)
    time_policy = _time_policy(float(speed_m_s), geometry, config)
    tolerance = 1.0e-8 * time_policy["flow_time_s"]
    if final_time < time_policy["first_checkpoint_time_s"] - tolerance:
        raise ValueError(f"completed transient case stopped before its first statistical checkpoint: {final_time:g} s")
    if final_time > time_policy["maximum_time_s"] + tolerance:
        raise ValueError(f"completed transient case exceeded its configured time ceiling: {final_time:g} s")

    force_root = validated_case / "postProcessing" / "forces"
    solver_info_root = validated_case / "postProcessing" / "solverInfo"
    force_history = parse_force_history(force_root)
    residuals = parse_solver_info(solver_info_root)
    statistics = select_converged_statistics(
        force_history,
        config,
        flow_time_s=time_policy["flow_time_s"],
        maximum_time_s=final_time,
    )
    residual_summary = summarize_residuals(
        residuals,
        config,
        window_start_time_s=float(force_history.times_s[statistics.start_index]),
        maximum_time_s=final_time,
    )
    wall_vtk = export_wall_fields(validated_case, runtime)
    yplus, cp_min = wall_field_statistics(wall_vtk)

    selected = slice(statistics.start_index, statistics.end_index)
    selected_times = force_history.times_s[selected]
    force_total = _time_average(selected_times, force_history.total_force_n[selected])
    force_pressure = _time_average(selected_times, force_history.pressure_force_n[selected])
    force_viscous = _time_average(selected_times, force_history.viscous_force_n[selected])
    moment = _time_average(selected_times, force_history.total_moment_nm[selected])
    drag = float(force_total[0])
    density = float(config.raw["fluid"]["density_kg_m3"])
    dynamic_pressure = 0.5 * density * float(speed_m_s) ** 2
    coefficient = drag / (dynamic_pressure * float(geometry.frontal_area_m2))
    hydrostatic = (
        float(config.raw["fluid"]["atmospheric_pressure_pa"])
        + density
        * float(config.raw["fluid"]["gravity_m_s2"])
        * float(config.raw["operating"]["centerline_depth_m"])
    )
    minimum_absolute_pressure = hydrostatic + cp_min * dynamic_pressure
    cavitation_margin = minimum_absolute_pressure - float(config.raw["fluid"]["vapour_pressure_pa"])

    solver_logs = sorted(validated_case.glob("log.pimpleFoam*"))
    if not solver_logs:
        raise ValueError(f"completed case lacks pimpleFoam logs: {validated_case}")
    maximum_courant = _maximum_courant_number(solver_logs)
    diagnostics = list(_solver_diagnostics(solver_logs))
    diagnostics.extend(statistics.reasons)
    if maximum_courant > float(config.raw["solver"]["maximum_courant_number"]):
        diagnostics.append("courant_number_limit")
    wall_policy = config.raw["solver"]
    if not (
        float(wall_policy["minimum_yplus_median"])
        <= yplus["median"]
        <= float(wall_policy["maximum_yplus_median"])
    ):
        diagnostics.append("yplus_median")
    if yplus["fraction_30_to_300"] < float(wall_policy["minimum_yplus_30_to_300_fraction"]):
        diagnostics.append("yplus_30_to_300_fraction")
    if yplus["fraction_above_300"] > float(wall_policy["preferred_yplus_above_300_fraction"]):
        diagnostics.append("warning_yplus_above_300_fraction")
    if yplus["fraction_above_300"] > float(wall_policy["maximum_yplus_above_300_fraction"]):
        diagnostics.append("yplus_above_300_fraction")
    diagnostics = list(dict.fromkeys(diagnostics))
    if any(code in diagnostics for code, _ in _SOLVER_FAILURE_PATTERNS) or "courant_number_limit" in diagnostics:
        rejection_code = "rejected_solver_divergence"
    elif not statistics.accepted:
        rejection_code = "rejected_statistical_convergence"
    elif any(code.startswith("yplus_") for code in diagnostics):
        rejection_code = "rejected_wall_resolution"
    else:
        rejection_code = None

    result = CaseResult(
        case_dir=validated_case,
        speed_m_s=float(speed_m_s),
        mesh_profile=mesh.profile,
        mesh_sha256=mesh.mesh_sha256,
        reynolds_number=float(speed_m_s) * geometry.length_m / float(config.raw["fluid"]["kinematic_viscosity_m2_s"]),
        accepted=rejection_code is None,
        rejection_code=rejection_code,
        diagnostic_codes=tuple(diagnostics),
        force_total_n=tuple(float(value) for value in force_total),
        force_pressure_n=tuple(float(value) for value in force_pressure),
        force_viscous_n=tuple(float(value) for value in force_viscous),
        moment_nm=tuple(float(value) for value in moment),
        drag_n=drag,
        drag_coefficient=coefficient,
        tow_power_w=drag * float(speed_m_s),
        force_stationary_mean_n=statistics.mean_drag_n,
        force_stationary_cov=statistics.drag_cov,
        force_mean_drift=statistics.drag_drift,
        force_stationary_start_time_s=float(force_history.times_s[statistics.start_index]),
        force_stationary_end_time_s=float(force_history.times_s[statistics.end_index - 1]),
        force_stationary_duration_flow_times=statistics.duration_s / time_policy["flow_time_s"],
        force_mean_uncertainty_n=statistics.mean_uncertainty_n,
        force_mean_uncertainty_fraction=statistics.mean_uncertainty_fraction,
        force_integral_time_scale_s=statistics.integral_time_scale_s,
        force_effective_sample_count=statistics.effective_sample_count,
        yplus=yplus,
        cp_min=cp_min,
        minimum_absolute_pressure_pa=minimum_absolute_pressure,
        cavitation_margin_pa=cavitation_margin,
        residuals=residual_summary,
        physical_time_s=final_time,
        time_steps=int(np.count_nonzero(force_history.times_s <= final_time + tolerance)),
        maximum_courant_number=maximum_courant,
        force_history_path=force_root,
        solver_info_path=solver_info_root,
        wall_fields_path=wall_vtk,
    )
    result_path = validated_case / "case-result.json"
    evidence_paths = [
        validated_case / "mesh-source.json",
        validated_case / "transient-protocol.json",
        validated_case / "log.decomposePar.solve",
        validated_case / "log.potentialFoam",
        validated_case / "log.reconstructPar.initial",
        *solver_logs,
        validated_case / "log.reconstructPar.final",
        validated_case / "log.yPlus.postProcess",
        validated_case / "log.foamToVTK.wall",
        wall_vtk,
        *sorted((validated_case / "convergence-checkpoints").glob("*.json")),
        *sorted(force_root.rglob("*.dat")),
        *sorted(solver_info_root.rglob("*.dat")),
    ]
    atomic_write_json(
        result_path,
        {
            "schema_version": 3,
            "result": case_result_payload(result),
            "evidence_sha256": _hashes(evidence_paths),
            "cavitation_model": "single-phase static-pressure screen at configured centreline depth",
            "time_averaging_policy": {
                "mode": "automatic_transient_then_ittc_running_mean",
                **time_policy,
                "completed_time_s": final_time,
                "completed_flow_through_times": final_time / time_policy["flow_time_s"],
                "checkpoint_times_s": list(_checkpoint_targets(time_policy)),
                "solver_module_sha256": sha256_file(Path(__file__).resolve()),
            },
        },
    )
    return result


def _protocol(case_dir: Path, speed_m_s: float, geometry: GeometryResult, mesh: MeshResult, config: PipelineConfig) -> dict[str, object]:
    policy = _time_policy(speed_m_s, geometry, config)
    payload: dict[str, object] = {
        "schema_version": 4,
        "solver": "pimpleFoam",
        "model": "URANS kOmegaSST",
        "initialization": "potentialFoam -initialiseUBCs -writephi -writep",
        "mesh_sha256": mesh.mesh_sha256,
        "speed_m_s": speed_m_s,
        "time_policy": policy,
        "checkpoint_times_s": list(_checkpoint_targets(policy)),
        "convergence": {
            "running_mean_uncertainty_fraction": float(
                config.raw["solver"]["drag_running_mean_uncertainty_fraction"]
            ),
            "mean_drift_fraction": float(config.raw["solver"]["drag_mean_drift_fraction"]),
            "transient_detection": "minimum uncertainty initial truncation over the first half",
            "mean_uncertainty": "ITTC late running-mean half range",
            "autocorrelation": "diagnostic only; not an acceptance gate",
        },
    }
    atomic_write_json(case_dir / "transient-protocol.json", payload)
    return payload


def run_case(
    case_dir: Path,
    speed_m_s: float,
    mesh: MeshResult,
    geometry: GeometryResult,
    runtime: OpenFoamRuntime,
    config: PipelineConfig,
) -> CaseResult:
    validated_case = Path(case_dir).resolve(strict=True)
    if not mesh.check_mesh_passed or not mesh.mesh_ok:
        raise ValueError(f"solver requires an accepted finite-volume mesh: {mesh.case_dir}")
    speed = float(speed_m_s)
    configured_speeds = tuple(float(value) for value in config.raw["operating"]["speeds_m_s"])
    if speed not in configured_speeds:
        raise ValueError(f"solver speed {speed:g} m/s is not in the configured sweep")
    ranks = int(config.raw["mpi_ranks"])
    _link_mesh(validated_case, mesh)
    protocol = _protocol(validated_case, speed, geometry, mesh, config)
    time_policy = protocol["time_policy"]
    assert isinstance(time_policy, dict)
    targets = tuple(float(value) for value in protocol["checkpoint_times_s"])  # type: ignore[arg-type]
    _run(
        [require_executable("decomposePar"), "-case", validated_case, "-force"],
        validated_case,
        "log.decomposePar.solve",
    )
    _run(
        _parallel_argv(
            runtime,
            ranks,
            require_executable("potentialFoam"),
            "-case",
            validated_case,
            "-parallel",
            "-initialiseUBCs",
            "-writephi",
            "-writep",
        ),
        validated_case,
        "log.potentialFoam",
    )
    _run(
        [require_executable("reconstructPar"), "-case", validated_case, "-latestTime", "-withZero"],
        validated_case,
        "log.reconstructPar.initial",
    )
    solver_logs: list[Path] = []
    for index, target in enumerate(targets):
        if index:
            _write_control_end_time(validated_case, target)
        flow_count = target / float(time_policy["flow_time_s"])
        solver_logs.append(
            _run(
                _parallel_argv(
                    runtime,
                    ranks,
                    runtime.pimple_foam,
                    "-case",
                    validated_case,
                    "-parallel",
                ),
                validated_case,
                f"log.pimpleFoam.to-{flow_count:.6g}Tf",
            )
        )
        final_time = _processor_latest_time(validated_case, ranks)
        if not math.isclose(final_time, target, rel_tol=0.0, abs_tol=1.0e-7 * float(time_policy["flow_time_s"])):
            raise ValueError(f"pimpleFoam ended at {final_time:g} s, expected checkpoint {target:g} s")
        converged, solver_failed = _write_convergence_checkpoint(
            validated_case,
            target,
            float(time_policy["flow_time_s"]),
            config,
            solver_logs,
        )
        if converged or solver_failed:
            break
    _run(
        [require_executable("reconstructPar"), "-case", validated_case, "-latestTime"],
        validated_case,
        "log.reconstructPar.final",
    )
    result = postprocess_completed_case(validated_case, speed, mesh, geometry, runtime, config)
    for processor in sorted(validated_case.glob("processor[0-9]*")):
        if processor.is_dir() and processor.parent.resolve() == validated_case:
            shutil.rmtree(processor)
    return result


def _validated_preserved_protocol(
    case_dir: Path,
    speed_m_s: float,
    mesh: MeshResult,
    geometry: GeometryResult,
    config: PipelineConfig,
) -> tuple[dict[str, object], dict[str, float]]:
    protocol_path = case_dir / "transient-protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 4 or protocol.get("solver") != "pimpleFoam":
        raise ValueError(f"preserved case does not use the current transient protocol: {case_dir}")
    if protocol.get("mesh_sha256") != mesh.mesh_sha256 or not math.isclose(
        float(protocol.get("speed_m_s", math.nan)), speed_m_s, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(f"preserved case protocol does not match mesh and speed: {case_dir}")
    expected = _time_policy(speed_m_s, geometry, config)
    recorded = protocol.get("time_policy")
    if not isinstance(recorded, dict) or any(
        not math.isclose(float(recorded.get(key, math.nan)), value, rel_tol=1.0e-12, abs_tol=1.0e-12)
        for key, value in expected.items()
    ):
        raise ValueError(f"preserved case time policy differs from current configuration: {case_dir}")
    return protocol, expected


def recover_converged_case(
    case_dir: Path,
    speed_m_s: float,
    mesh: MeshResult,
    geometry: GeometryResult,
    runtime: OpenFoamRuntime,
    config: PipelineConfig,
) -> CaseResult:
    validated_case = Path(case_dir).resolve(strict=True)
    _, time_policy = _validated_preserved_protocol(validated_case, float(speed_m_s), mesh, geometry, config)
    final_time = _processor_latest_time(validated_case, int(config.raw["mpi_ranks"]))
    solver_logs = sorted(validated_case.glob("log.pimpleFoam*"))
    converged, solver_failed = _write_convergence_checkpoint(
        validated_case,
        final_time,
        time_policy["flow_time_s"],
        config,
        solver_logs,
    )
    if solver_failed or not converged:
        raise ValueError(f"preserved transient case is not statistically converged at {final_time:g} s")
    _run(
        [require_executable("reconstructPar"), "-case", validated_case, "-latestTime"],
        validated_case,
        "log.reconstructPar.final",
    )
    result = postprocess_completed_case(validated_case, speed_m_s, mesh, geometry, runtime, config)
    if not result.accepted:
        raise ValueError(f"preserved solver case failed post-processing acceptance: {validated_case}")
    for processor in sorted(validated_case.glob("processor[0-9]*")):
        if processor.is_dir() and processor.parent.resolve() == validated_case:
            shutil.rmtree(processor)
    return result


def resume_preserved_case(
    case_dir: Path,
    speed_m_s: float,
    mesh: MeshResult,
    geometry: GeometryResult,
    runtime: OpenFoamRuntime,
    config: PipelineConfig,
) -> CaseResult:
    validated_case = Path(case_dir).resolve(strict=True)
    protocol, time_policy = _validated_preserved_protocol(validated_case, float(speed_m_s), mesh, geometry, config)
    ranks = int(config.raw["mpi_ranks"])
    current = _processor_latest_time(validated_case, ranks)
    targets = tuple(float(value) for value in protocol["checkpoint_times_s"])  # type: ignore[arg-type]
    remaining = tuple(target for target in targets if target > current + 1.0e-8 * time_policy["flow_time_s"])
    if not remaining:
        raise ValueError(f"preserved transient case has no remaining statistical checkpoint: {validated_case}")
    solver_logs = sorted(validated_case.glob("log.pimpleFoam*"))
    if not solver_logs or _solver_diagnostics(solver_logs):
        raise ValueError(f"preserved transient case lacks clean pimpleFoam history: {validated_case}")
    for target in remaining:
        _write_control_end_time(validated_case, target)
        flow_count = target / time_policy["flow_time_s"]
        continuation = _run(
            _parallel_argv(
                runtime,
                ranks,
                runtime.pimple_foam,
                "-case",
                validated_case,
                "-parallel",
            ),
            validated_case,
            f"log.pimpleFoam.resume-to-{flow_count:.6g}Tf",
        )
        solver_logs.append(continuation)
        final_time = _processor_latest_time(validated_case, ranks)
        if not math.isclose(final_time, target, rel_tol=0.0, abs_tol=1.0e-7 * time_policy["flow_time_s"]):
            raise ValueError(f"resumed pimpleFoam ended at {final_time:g} s, expected {target:g} s")
        converged, solver_failed = _write_convergence_checkpoint(
            validated_case,
            target,
            time_policy["flow_time_s"],
            config,
            solver_logs,
        )
        if converged or solver_failed:
            break
    _run(
        [require_executable("reconstructPar"), "-case", validated_case, "-latestTime"],
        validated_case,
        "log.reconstructPar.final",
    )
    result = postprocess_completed_case(validated_case, speed_m_s, mesh, geometry, runtime, config)
    for processor in sorted(validated_case.glob("processor[0-9]*")):
        if processor.is_dir() and processor.parent.resolve() == validated_case:
            shutil.rmtree(processor)
    return result


def write_task_05_checkpoint(project_root: Path, result: CaseResult) -> Path:
    result_path = result.case_dir / "case-result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = project_root / "docs" / "superpowers" / "plans" / "evidence" / "task-05-case.json"
    atomic_write_json(
        checkpoint,
        {
            "schema_version": 2,
            "case_result": payload,
            "case_result_sha256": sha256_file(result_path),
            "solver_log_hashes": _hashes(sorted(result.case_dir.glob("log.pimpleFoam*"))),
            "force_file_hashes": _hashes(sorted(result.force_history_path.rglob("*.dat"))),
            "wall_field_export_sha256": sha256_file(result.wall_fields_path),
        },
    )
    return checkpoint


def write_case_rejection(case_dir: Path, error: BaseException) -> Path:
    code = "command_failed" if isinstance(error, CommandFailure) else "solver_pipeline_error"
    payload: dict[str, object] = {
        "schema_version": 2,
        "accepted": False,
        "rejection_code": code,
        "error": str(error),
    }
    if isinstance(error, CommandFailure):
        payload["argv"] = list(error.argv)
        payload["returncode"] = error.returncode
        payload["log"] = str(error.log_path)
    path = case_dir / "case-rejection.json"
    atomic_write_json(path, payload)
    return path
