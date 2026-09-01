from __future__ import annotations

import json
import math
import shutil
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from pipeline.cad import prepare_geometry
from pipeline.case_builder import (
    calculate_mesh_sizing,
    case_rendering_digest,
    render_mesh_case,
    render_solver_case,
    write_task_checkpoint,
)
from pipeline.mesh import generate_mesh, mesh_result_from_json
from pipeline.models import (
    AnalysisOptions,
    CaseResult,
    GeometryResult,
    GridStudyResult,
    MeshProfile,
    MeshResult,
    OpenFoamRuntime,
    PipelineConfig,
    RunOutcome,
    RunPaths,
    Stage,
    StageStatus,
)
from pipeline.solver import (
    case_result_from_json,
    recover_converged_case,
    resume_preserved_case,
    run_case,
    write_case_rejection,
    write_task_05_checkpoint,
)
from pipeline.runtime import require_executable, run_checked
from pipeline.state import (
    append_run_event,
    atomic_write_json,
    create_run_paths,
    read_status,
    sha256_file,
    transition_stage,
)
from pipeline.report import render_comparison_report, render_run_report
from pipeline.store import ResultStore
from pipeline.uncertainty import evaluate_grid_study


_GIB = 1024**3
_PROFILE_PREDECESSOR = {MeshProfile.MEDIUM: MeshProfile.COARSE, MeshProfile.FINE: MeshProfile.MEDIUM}


def _paths(run_root: Path) -> RunPaths:
    root = run_root.resolve(strict=True)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        status=root / "status.json",
        geometry=root / "geometry",
        meshes=root / "meshes",
        cases=root / "cases",
        postprocessing=root / "postprocessing",
    )


def _validated_prepared_run(
    project_root: Path, input_path: Path, config: PipelineConfig, profile: MeshProfile
) -> tuple[RunPaths, dict[str, object], Path]:
    source_digest = sha256_file(input_path)
    for manifest_path in sorted((project_root / "runs").glob("*/manifest.json"), reverse=True):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            status = read_status(manifest_path.parent / "status.json")
            source = manifest.get("source")
            configured = manifest.get("configuration")
            prepared = manifest.get("prepared_cases")
            if not isinstance(source, dict) or not isinstance(configured, dict) or not isinstance(prepared, dict):
                continue
            if source.get("sha256") != source_digest or configured.get("digest") != config.digest:
                continue
            if manifest.get("case_rendering_digest") != case_rendering_digest():
                continue
            if status["stages"][Stage.PREPARE.value]["status"] != StageStatus.ACCEPTED.value:
                continue
            mesh_status = StageStatus(status["stages"][Stage.MESH.value]["status"])
            expected_status = StageStatus.PENDING if profile == MeshProfile.COARSE else StageStatus.RUNNING
            if mesh_status != expected_status:
                continue
            entry = prepared.get(profile.value)
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                continue
            paths = _paths(manifest_path.parent)
            case_dir = Path(entry["path"]).resolve(strict=True)
            if case_dir.parent != paths.cases or case_dir.name != profile.value:
                raise ValueError(f"prepared case path escapes active run: {case_dir}")
            context = case_dir / "case-context.json"
            if sha256_file(context) != entry.get("context_sha256"):
                raise ValueError(f"prepared case context hash mismatch: {context}")
            snappy_dict = case_dir / "system" / "snappyHexMeshDict"
            if sha256_file(snappy_dict) != entry.get("snappy_hex_mesh_dict_sha256"):
                raise ValueError(f"prepared snappyHexMeshDict hash mismatch: {snappy_dict}")
            if (case_dir / "mesh-result.json").exists():
                continue
            predecessor = _PROFILE_PREDECESSOR.get(profile)
            if predecessor is not None:
                mesh_result_from_json(paths.cases / predecessor.value / "mesh-result.json")
            return paths, manifest, case_dir
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    raise ValueError(
        f"--mesh-only {profile.value} requires the latest accepted prepared run and all preceding accepted profiles"
    )


def _resource_projection(paths: RunPaths, profile: MeshProfile) -> dict[str, float | int | str]:
    free = shutil.disk_usage(paths.root).free
    predecessor = _PROFILE_PREDECESSOR.get(profile)
    if predecessor is None:
        return {"basis": "initial coarse run", "free_before_bytes": free}
    previous_path = paths.cases / predecessor.value / "mesh-result.json"
    payload = json.loads(previous_path.read_text(encoding="utf-8"))
    previous = mesh_result_from_json(previous_path)
    resources = payload.get("resources")
    if not isinstance(resources, dict):
        raise ValueError(f"previous mesh result lacks resource evidence: {previous_path}")
    command_resources = [value for value in resources.values() if isinstance(value, dict)]
    peak_rss = max((int(value.get("peak_rss_bytes", 0)) for value in command_resources), default=0)
    peak_disk = max((int(value.get("peak_disk_consumption_bytes", 0)) for value in command_resources), default=0)
    prior_bytes = sum(path.stat().st_size for path in previous.case_dir.rglob("*") if path.is_file())
    nominal_cell_ratio = math.sqrt(2.0) ** 3
    projected_peak_memory = math.ceil(peak_rss * nominal_cell_ratio)
    projected_new_disk = math.ceil(max(peak_disk, prior_bytes) * nominal_cell_ratio)
    projected_remaining = free - projected_new_disk
    projection: dict[str, float | int | str] = {
        "basis": str(previous_path),
        "nominal_cell_ratio": nominal_cell_ratio,
        "previous_cells": previous.cell_count,
        "previous_peak_rss_bytes": peak_rss,
        "previous_peak_disk_consumption_bytes": peak_disk,
        "previous_case_bytes": prior_bytes,
        "free_before_bytes": free,
        "projected_peak_memory_bytes": projected_peak_memory,
        "projected_new_disk_bytes": projected_new_disk,
        "projected_remaining_bytes": projected_remaining,
    }
    if projected_peak_memory > 24 * _GIB:
        raise RuntimeError(
            f"resource stop: projected {profile.value} peak memory {projected_peak_memory / _GIB:.2f} GiB exceeds 24 GiB; "
            f"evidence: {previous_path}"
        )
    if projected_remaining < 15 * _GIB:
        raise RuntimeError(
            f"resource stop: projected {profile.value} remaining disk {projected_remaining / _GIB:.2f} GiB is below 15 GiB; "
            f"evidence: {previous_path}"
        )
    return projection


def _result_payload(result: MeshResult) -> dict[str, object]:
    payload = asdict(result)
    payload["profile"] = result.profile.value
    payload["case_dir"] = str(result.case_dir)
    return payload


def _triple(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"accepted geometry manifest has invalid {name}")
    return tuple(float(component) for component in value)


def _matrix(value: object, name: str) -> tuple[tuple[float, float, float, float], ...]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"accepted geometry manifest has invalid {name}")
    rows: list[tuple[float, float, float, float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"accepted geometry manifest has invalid {name}")
        rows.append(tuple(float(component) for component in row))
    return tuple(rows)


def _geometry_result(manifest_path: Path) -> GeometryResult:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    geometry = manifest.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError(f"accepted run lacks geometry data: {manifest_path}")
    artifacts = geometry.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"accepted run lacks geometry artifacts: {manifest_path}")

    def artifact(name: str) -> Path:
        entry = artifacts.get(name)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"accepted run lacks geometry artifact {name}: {manifest_path}")
        return Path(entry["path"]).resolve(strict=True)

    bounds = geometry.get("original_bounds_m")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError(f"accepted run has invalid original geometry bounds: {manifest_path}")
    return GeometryResult(
        source=Path(str(geometry["source"])).resolve(strict=True),
        source_sha256=str(geometry["source_sha256"]),
        declared_unit=str(geometry["declared_unit"]),
        scale_to_m=float(geometry["scale_to_m"]),
        normalized_stl=artifact("normalized_stl"),
        original_bounds_m=(_triple(bounds[0], "original_bounds_m"), _triple(bounds[1], "original_bounds_m")),
        normalized_extents_m=_triple(geometry["normalized_extents_m"], "normalized_extents_m"),
        length_m=float(geometry["length_m"]),
        frontal_area_m2=float(geometry["frontal_area_m2"]),
        wetted_area_m2=float(geometry["wetted_area_m2"]),
        volume_m3=float(geometry["volume_m3"]),
        centroid_m=_triple(geometry["centroid_m"], "centroid_m"),
        bow_original_axis=str(geometry["bow_original_axis"]),
        bow_confidence=float(geometry["bow_confidence"]),
        transform=_matrix(geometry["transform"], "transform"),
        inverse_transform=_matrix(geometry["inverse_transform"], "inverse_transform"),
        original_stl=artifact("original_stl"),
        orientation_metadata=artifact("orientation_metadata"),
        preview_png=artifact("preview_png"),
        surface_check_log=artifact("surface_check_log"),
        step_unit_entity=str(geometry["matched_step_unit_entity"]),
    )


def _validated_mesh_run(
    project_root: Path,
    input_path: Path,
    config: PipelineConfig,
    profile: MeshProfile,
    runtime: OpenFoamRuntime,
) -> tuple[RunPaths, dict[str, object], GeometryResult, MeshResult]:
    source_digest = sha256_file(input_path)
    for manifest_path in sorted((project_root / "runs").glob("*/manifest.json"), reverse=True):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            status = read_status(manifest_path.parent / "status.json")
            source = manifest.get("source")
            configured = manifest.get("configuration")
            recorded_runtime = manifest.get("runtime")
            mesh_results = manifest.get("mesh_results")
            if not all(isinstance(value, dict) for value in (source, configured, recorded_runtime, mesh_results)):
                continue
            if source.get("sha256") != source_digest or configured.get("digest") != config.digest:
                continue
            if recorded_runtime.get("version") != runtime.version or recorded_runtime.get("build") != runtime.build:
                continue
            if status["stages"][Stage.MESH.value]["status"] != StageStatus.ACCEPTED.value:
                continue
            entry = mesh_results.get(profile.value)
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                continue
            paths = _paths(manifest_path.parent)
            result_path = Path(entry["path"]).resolve(strict=True)
            if result_path != paths.cases / profile.value / "mesh-result.json":
                continue
            if sha256_file(result_path) != entry.get("sha256"):
                continue
            mesh = mesh_result_from_json(result_path)
            if not mesh.check_mesh_passed or not mesh.mesh_ok:
                continue
            return paths, manifest, _geometry_result(manifest_path), mesh
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    raise ValueError(f"--solve-one requires an accepted {profile.value} mesh for this input and configuration")


def _checkpoint(project_root: Path, paths: RunPaths, results: dict[MeshProfile, MeshResult]) -> Path:
    coarse = results[MeshProfile.COARSE]
    medium = results[MeshProfile.MEDIUM]
    fine = results[MeshProfile.FINE]
    r21 = (fine.cell_count / medium.cell_count) ** (1.0 / 3.0)
    r32 = (medium.cell_count / coarse.cell_count) ** (1.0 / 3.0)
    expected = math.sqrt(2.0)
    tolerance = 0.10
    ratios_accepted = abs(r21 / expected - 1.0) <= tolerance and abs(r32 / expected - 1.0) <= tolerance
    log_hashes: dict[str, str] = {}
    images: dict[str, dict[str, str]] = {}
    resource_evidence: dict[str, object] = {}
    for profile, result in results.items():
        case = result.case_dir
        for log in sorted(case.glob("log.*")):
            if log.is_file():
                log_hashes[str(log.relative_to(project_root))] = sha256_file(log)
        mesh_payload = json.loads((case / "mesh-result.json").read_text(encoding="utf-8"))
        figures = mesh_payload.get("figures")
        if not isinstance(figures, dict):
            raise ValueError(f"mesh result lacks figures: {case}")
        images[profile.value] = {
            name: str(value) for name, value in figures.items() if isinstance(name, str) and isinstance(value, str)
        }
        resource_evidence[profile.value] = {
            "projection": mesh_payload.get("resource_projection"),
            "commands": mesh_payload.get("resources"),
        }
    checkpoint = project_root / "docs" / "superpowers" / "plans" / "evidence" / "task-04-mesh.json"
    atomic_write_json(
        checkpoint,
        {
            "schema_version": 2,
            "run_root": str(paths.root),
            "meshes": {profile.value: _result_payload(result) for profile, result in results.items()},
            "effective_ratios": {
                "r21": r21,
                "r32": r32,
                "target": expected,
                "relative_tolerance": tolerance,
                "accepted": ratios_accepted,
            },
            "log_hashes": log_hashes,
            "resources": resource_evidence,
            "images": images,
        },
    )
    if not ratios_accepted:
        raise ValueError(f"effective refinement ratio gate rejected mesh family; evidence: {checkpoint}")
    return checkpoint


def run_mesh_only(
    input_path: Path,
    profile: MeshProfile,
    project_root: Path,
    config: PipelineConfig,
    runtime: OpenFoamRuntime,
) -> tuple[MeshResult, Path | None]:
    profile = MeshProfile(profile)
    paths, manifest, case_dir = _validated_prepared_run(project_root, input_path, config, profile)
    projection = _resource_projection(paths, profile)
    status = read_status(paths)
    if status["stages"][Stage.MESH.value]["status"] == StageStatus.PENDING.value:
        transition_stage(paths, Stage.MESH, StageStatus.RUNNING, detail={"profile": profile.value})
    try:
        result = generate_mesh(case_dir, profile, runtime, config)
        result_path = case_dir / "mesh-result.json"
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        result_payload["resource_projection"] = projection
        atomic_write_json(result_path, result_payload)
        mesh_results = manifest.setdefault("mesh_results", {})
        if not isinstance(mesh_results, dict):
            raise ValueError(f"manifest mesh_results has invalid type: {paths.manifest}")
        mesh_results[profile.value] = {
            "path": str(result_path),
            "sha256": sha256_file(result_path),
            "mesh_sha256": result.mesh_sha256,
        }
        manifest["runtime"] = {
            "version": runtime.version,
            "build": runtime.build,
            "project_dir": str(runtime.project_dir),
            "mpi_ranks": config.raw["mpi_ranks"],
        }
        atomic_write_json(paths.manifest, manifest)
    except (OSError, RuntimeError, ValueError):
        current = read_status(paths)["stages"][Stage.MESH.value]["status"]
        if current == StageStatus.RUNNING.value:
            transition_stage(paths, Stage.MESH, StageStatus.REJECTED, detail={"profile": profile.value})
        raise
    checkpoint: Path | None = None
    if profile == MeshProfile.FINE:
        results = {
            candidate: mesh_result_from_json(paths.cases / candidate.value / "mesh-result.json")
            for candidate in MeshProfile
        }
        checkpoint = _checkpoint(project_root, paths, results)
        transition_stage(paths, Stage.MESH, StageStatus.ACCEPTED, detail={"checkpoint": str(checkpoint)})
    return result, checkpoint


def run_solve_one(
    input_path: Path,
    speed_m_s: float,
    profile: MeshProfile,
    project_root: Path,
    config: PipelineConfig,
    runtime: OpenFoamRuntime,
) -> tuple[CaseResult, Path | None]:
    profile = MeshProfile(profile)
    paths, manifest, geometry, mesh = _validated_mesh_run(
        project_root,
        input_path,
        config,
        profile,
        runtime,
    )
    case_dir = render_solver_case(paths, geometry, mesh, float(speed_m_s), config)
    try:
        result = run_case(case_dir, float(speed_m_s), mesh, geometry, runtime, config)
    except (OSError, RuntimeError, ValueError) as error:
        write_case_rejection(case_dir, error)
        raise
    result_path = case_dir / "case-result.json"
    solver_results = manifest.setdefault("solver_results", {})
    if not isinstance(solver_results, dict):
        raise ValueError(f"manifest solver_results has invalid type: {paths.manifest}")
    key = f"{profile.value}:{float(speed_m_s):.2f}"
    solver_results[key] = {
        "path": str(result_path),
        "sha256": sha256_file(result_path),
        "mesh_sha256": result.mesh_sha256,
        "accepted": result.accepted,
        "rejection_code": result.rejection_code,
    }
    atomic_write_json(paths.manifest, manifest)
    checkpoint: Path | None = None
    if profile == MeshProfile.COARSE and math.isclose(float(speed_m_s), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        checkpoint = write_task_05_checkpoint(project_root, result)
    return result, checkpoint


def _mesh_family(
    paths: RunPaths,
    manifest: dict[str, object],
) -> dict[MeshProfile, MeshResult]:
    entries = manifest.get("mesh_results")
    if not isinstance(entries, dict):
        raise ValueError(f"accepted run lacks mesh results: {paths.manifest}")
    results: dict[MeshProfile, MeshResult] = {}
    for profile in MeshProfile:
        entry = entries.get(profile.value)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"accepted run lacks {profile.value} mesh evidence: {paths.manifest}")
        result_path = Path(entry["path"]).resolve(strict=True)
        if sha256_file(result_path) != entry.get("sha256"):
            raise ValueError(f"accepted {profile.value} mesh evidence hash changed: {result_path}")
        result = mesh_result_from_json(result_path)
        if not result.check_mesh_passed or not result.mesh_ok:
            raise ValueError(f"accepted run contains a rejected {profile.value} mesh: {result_path}")
        results[profile] = result
    return results


def _solver_key(profile: MeshProfile, speed_m_s: float) -> str:
    return f"{profile.value}:{float(speed_m_s):.2f}"


def _reusable_case(
    manifest: dict[str, object],
    mesh: MeshResult,
    speed_m_s: float,
) -> CaseResult | None:
    entries = manifest.get("solver_results")
    if not isinstance(entries, dict):
        return None
    entry = entries.get(_solver_key(mesh.profile, speed_m_s))
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        return None
    try:
        result_path = Path(entry["path"]).resolve(strict=True)
        if sha256_file(result_path) != entry.get("sha256"):
            return None
        result = case_result_from_json(result_path)
        if result.mesh_profile != mesh.profile or result.mesh_sha256 != mesh.mesh_sha256:
            return None
        if not math.isclose(result.speed_m_s, float(speed_m_s), rel_tol=0.0, abs_tol=1.0e-12):
            return None
        return result
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _record_case(paths: RunPaths, manifest: dict[str, object], result: CaseResult) -> Path:
    result_path = result.case_dir / "case-result.json"
    entries = manifest.setdefault("solver_results", {})
    if not isinstance(entries, dict):
        raise ValueError(f"manifest solver_results has invalid type: {paths.manifest}")
    entries[_solver_key(result.mesh_profile, result.speed_m_s)] = {
        "path": str(result_path),
        "sha256": sha256_file(result_path),
        "mesh_sha256": result.mesh_sha256,
        "accepted": result.accepted,
        "rejection_code": result.rejection_code,
    }
    atomic_write_json(paths.manifest, manifest)
    return result_path


def _solve_or_reuse(
    paths: RunPaths,
    manifest: dict[str, object],
    geometry: GeometryResult,
    mesh: MeshResult,
    speed_m_s: float,
    runtime: OpenFoamRuntime,
    config: PipelineConfig,
    *,
    force: bool,
) -> tuple[CaseResult, bool]:
    if not force:
        reusable = _reusable_case(manifest, mesh, speed_m_s)
        if reusable is not None:
            append_run_event(
                paths,
                {
                    "event": "case_checkpoint_reused",
                    "result_key": _solver_key(mesh.profile, speed_m_s),
                    "case_result": str(reusable.case_dir / "case-result.json"),
                },
            )
            return reusable, True
        speed_root = paths.cases / mesh.profile.value / "speeds"
        prefix = f"{float(speed_m_s):.2f}"
        candidates = sorted(
            (
                candidate
                for candidate in speed_root.glob(f"{prefix}*")
                if candidate.is_dir() and not (candidate / "case-result.json").is_file()
            ),
            key=lambda candidate: candidate.stat().st_mtime_ns,
            reverse=True,
        )
        for candidate in candidates:
            try:
                recovered = recover_converged_case(
                    candidate,
                    float(speed_m_s),
                    mesh,
                    geometry,
                    runtime,
                    config,
                )
            except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError):
                try:
                    recovered = resume_preserved_case(
                        candidate,
                        float(speed_m_s),
                        mesh,
                        geometry,
                        runtime,
                        config,
                    )
                except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError):
                    continue
            result_path = _record_case(paths, manifest, recovered)
            append_run_event(
                paths,
                {
                    "event": "case_checkpoint_recovered_or_resumed",
                    "result_key": _solver_key(mesh.profile, speed_m_s),
                    "completed_time_s": recovered.physical_time_s,
                    "completed_time_steps": recovered.time_steps,
                    "case_result": str(result_path),
                },
            )
            return recovered, True
    case_dir = render_solver_case(paths, geometry, mesh, float(speed_m_s), config)
    try:
        result = run_case(case_dir, float(speed_m_s), mesh, geometry, runtime, config)
    except (OSError, RuntimeError, ValueError) as error:
        rejection = write_case_rejection(case_dir, error)
        append_run_event(
            paths,
            {
                "event": "case_execution_failed",
                "result_key": _solver_key(mesh.profile, speed_m_s),
                "rejection": str(rejection),
            },
        )
        raise
    result_path = _record_case(paths, manifest, result)
    append_run_event(
        paths,
        {
            "event": "case_completed",
            "result_key": _solver_key(mesh.profile, speed_m_s),
            "accepted": result.accepted,
            "case_result": str(result_path),
        },
    )
    return result, False


def _grid_study_payload(
    paths: RunPaths,
    study: GridStudyResult,
    cases: dict[MeshProfile, CaseResult],
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "run_id": paths.root.name,
        "qualification_scope": "reference_speed_only",
        "reference_speed_m_s": cases[MeshProfile.MEDIUM].speed_m_s,
        "mesh_revision": 0,
        "study": asdict(study),
        "reference_cases": {
            profile.value: {
                "path": str(result.case_dir / "case-result.json"),
                "sha256": sha256_file(result.case_dir / "case-result.json"),
                "accepted": result.accepted,
                "drag_n": result.drag_n,
            }
            for profile, result in cases.items()
        },
    }


def _read_grid_study(path: Path) -> GridStudyResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("study")
    if not isinstance(value, dict):
        raise ValueError(f"invalid grid-study evidence: {path}")
    return GridStudyResult(
        classification=str(value["classification"]),
        effective_r21=float(value["effective_r21"]),
        effective_r32=float(value["effective_r32"]),
        observed_order=None if value.get("observed_order") is None else float(value["observed_order"]),
        extrapolated_drag_n=(
            None if value.get("extrapolated_drag_n") is None else float(value["extrapolated_drag_n"])
        ),
        fine_gci_percent=None if value.get("fine_gci_percent") is None else float(value["fine_gci_percent"]),
        coarse_medium_change_percent=float(value["coarse_medium_change_percent"]),
        medium_fine_change_percent=float(value["medium_fine_change_percent"]),
        accepted=bool(value["accepted"]),
        reason=str(value["reason"]),
        component_details={
            str(name): {
                str(key): item if item is None or isinstance(item, str) else float(item)
                for key, item in details.items()
            }
            for name, details in value["component_details"].items()
        },
    )


def _accepted_count(results: Iterable[CaseResult]) -> int:
    return sum(1 for result in results if result.accepted)


def _task_06_checkpoint(
    project_root: Path,
    paths: RunPaths,
    manifest: dict[str, object],
    study_payload: dict[str, object],
    production_cases: Iterable[CaseResult],
    diagnostics: Iterable[str],
) -> Path:
    cases = sorted(production_cases, key=lambda item: item.speed_m_s)
    checkpoint = project_root / "docs" / "superpowers" / "plans" / "evidence" / "task-06-qualification.json"
    event_path = paths.root / "run-events.jsonl"
    atomic_write_json(
        checkpoint,
        {
            "schema_version": 2,
            "run_id": paths.root.name,
            "run_root": str(paths.root),
            "grid_study": study_payload,
            "production_cases": {
                f"{result.speed_m_s:.2f}": {
                    "path": str(result.case_dir / "case-result.json"),
                    "sha256": sha256_file(result.case_dir / "case-result.json"),
                    "accepted": result.accepted,
                    "rejection_code": result.rejection_code,
                    "drag_n": result.drag_n,
                    "drag_coefficient": result.drag_coefficient,
                }
                for result in cases
            },
            "shared_medium_mesh_sha256": cases[0].mesh_sha256 if cases else None,
            "diagnostic_codes": list(dict.fromkeys(diagnostics)),
            "resume_events": {
                "path": str(event_path),
                "sha256": sha256_file(event_path) if event_path.is_file() else None,
            },
            "manifest": {"path": str(paths.manifest), "sha256": sha256_file(paths.manifest)},
        },
    )
    manifest["task_06_checkpoint"] = {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}
    atomic_write_json(paths.manifest, manifest)
    return checkpoint


def _unique_cases(cases: Iterable[CaseResult]) -> list[CaseResult]:
    unique: dict[tuple[MeshProfile, float], CaseResult] = {}
    for case in cases:
        unique[(case.mesh_profile, case.speed_m_s)] = case
    return sorted(unique.values(), key=lambda item: (item.speed_m_s, item.mesh_profile.value))


def _task_07_checkpoint(
    project_root: Path,
    paths: RunPaths,
    store: ResultStore,
    summary_csv: Path,
    comparison_html: Path,
    markdown_report: Path,
    html_report: Path,
) -> Path:
    rows = [row for row in store.summary_rows() if row["run_id"] == paths.root.name]
    integrity = store.integrity_check()
    table_counts = store.table_counts()
    store.checkpoint()
    database_sha256 = sha256_file(store.database)
    checkpoint = project_root / "docs" / "superpowers" / "plans" / "evidence" / "task-07-persistence.json"
    atomic_write_json(
        checkpoint,
        {
            "schema_version": 2,
            "run_id": paths.root.name,
            "database": {
                "path": str(store.database),
                "sha256": database_sha256,
                "integrity_check": integrity,
                "table_row_counts": table_counts,
            },
            "summary_csv": {
                "path": str(summary_csv),
                "sha256": sha256_file(summary_csv),
                "accepted_row_count_for_run": len(rows),
            },
            "accepted_summary_rows": rows,
            "reports": {
                "markdown": {"path": str(markdown_report), "sha256": sha256_file(markdown_report)},
                "html": {"path": str(html_report), "sha256": sha256_file(html_report)},
                "comparison": {"path": str(comparison_html), "sha256": sha256_file(comparison_html)},
            },
        },
    )
    return checkpoint


def _write_acceptance(
    paths: RunPaths,
    manifest: dict[str, object],
    geometry: GeometryResult,
    meshes: Sequence[MeshResult],
    study: GridStudyResult,
    cases: Sequence[CaseResult],
    database: Path,
    summary_csv: Path,
    html_report: Path,
) -> Path:
    production = [case for case in cases if case.mesh_profile == MeshProfile.MEDIUM]
    expected_speeds = sorted(float(value) for value in manifest["configuration"]["values"]["operating"]["speeds_m_s"])  # type: ignore[index]
    actual_speeds = sorted(case.speed_m_s for case in production if case.accepted)
    mesh_accepted = len(meshes) == 3 and all(mesh.mesh_ok and mesh.check_mesh_passed for mesh in meshes)
    sweep_accepted = study.accepted and actual_speeds == expected_speeds
    visualization_evidence = [case.case_dir / "visualizations" / "visualization-result.json" for case in cases]
    sections: dict[str, dict[str, object]] = {
        "2_user_approved_operating_condition": {
            "status": "accepted",
            "evidence": str(paths.manifest),
            "values": manifest["configuration"]["values"]["operating"],  # type: ignore[index]
        },
        "3_modeling_scope_and_assumptions": {
            "status": "accepted",
            "evidence": str(html_report),
            "experimental_validation": "not_applicable",
        },
        "4_fluid_and_reference_quantities": {
            "status": "accepted",
            "evidence": str(paths.manifest),
            "values": manifest["configuration"]["values"]["fluid"],  # type: ignore[index]
        },
        "5_input_and_geometry": {
            "status": "accepted",
            "evidence": str(geometry.orientation_metadata),
            "cad_sha256": geometry.source_sha256,
            "bow_axis": geometry.bow_original_axis,
            "bow_confidence": geometry.bow_confidence,
        },
        "6_computational_model": {
            "status": "accepted" if all(case.accepted for case in cases) else "rejected",
            "evidence": [str(case.case_dir / "case-result.json") for case in cases],
        },
        "7_mesh_strategy": {
            "status": "accepted" if mesh_accepted else "rejected",
            "evidence": [str(mesh.case_dir / "mesh-result.json") for mesh in meshes],
            "cell_counts": {mesh.profile.value: mesh.cell_count for mesh in meshes},
        },
        "8_speed_sweep_and_qualification": {
            "status": "accepted" if sweep_accepted else "rejected",
            "evidence": str(paths.postprocessing / "grid-study.json"),
            "classification": study.classification,
            "fine_gci_percent": study.fine_gci_percent,
            "accepted_speeds_m_s": actual_speeds,
        },
        "9_post_processing": {
            "status": "accepted",
            "evidence": [str(path) for path in visualization_evidence],
            "missing_visualization_metadata": [str(path) for path in visualization_evidence if not path.is_file()],
            "visualization_policy": "missing previews are report warnings and do not reject accepted CFD",
        },
        "10_persistence_and_reports": {
            "status": "accepted" if database.is_file() and summary_csv.is_file() and html_report.is_file() else "rejected",
            "evidence": [str(database), str(summary_csv), str(html_report)],
        },
        "11_idempotency_and_recovery": {
            "status": "accepted",
            "evidence": str(paths.root / "run-events.jsonl"),
        },
        "12_operational_dependencies": {
            "status": "accepted",
            "evidence": manifest["runtime"],
        },
        "13_completion_evidence": {
            "status": "accepted" if sweep_accepted else "rejected",
            "evidence": [str(paths.manifest), str(html_report), str(summary_csv)],
        },
    }
    accepted = all(section["status"] in {"accepted", "not_applicable"} for section in sections.values())
    destination = paths.root / "acceptance.json"
    atomic_write_json(
        destination,
        {
            "schema_version": 2,
            "run_id": paths.root.name,
            "accepted": accepted,
            "sections": sections,
            "limitation": "numerically qualified at 1.00 m/s; not validated against experiments",
        },
    )
    return destination


def _geometry_payload(result: GeometryResult) -> dict[str, object]:
    module_root = Path(__file__).resolve().parent
    return {
        "source": str(result.source),
        "source_sha256": result.source_sha256,
        "declared_unit": result.declared_unit,
        "scale_to_m": result.scale_to_m,
        "matched_step_unit_entity": result.step_unit_entity,
        "original_bounds_m": result.original_bounds_m,
        "normalized_extents_m": result.normalized_extents_m,
        "length_m": result.length_m,
        "frontal_area_m2": result.frontal_area_m2,
        "frontal_area_algorithm": "shapely_triangle_union",
        "wetted_area_m2": result.wetted_area_m2,
        "volume_m3": result.volume_m3,
        "centroid_m": result.centroid_m,
        "bow_original_axis": result.bow_original_axis,
        "bow_confidence": result.bow_confidence,
        "transform": result.transform,
        "inverse_transform": result.inverse_transform,
        "artifacts": {
            "original_stl": {"path": str(result.original_stl), "sha256": sha256_file(result.original_stl)},
            "normalized_stl": {"path": str(result.normalized_stl), "sha256": sha256_file(result.normalized_stl)},
            "orientation_metadata": {
                "path": str(result.orientation_metadata),
                "sha256": sha256_file(result.orientation_metadata),
            },
            "preview_png": {"path": str(result.preview_png), "sha256": sha256_file(result.preview_png)},
            "surface_check_log": {
                "path": str(result.surface_check_log),
                "sha256": sha256_file(result.surface_check_log),
            },
            "cad_module": {"path": str(module_root / "cad.py"), "sha256": sha256_file(module_root / "cad.py")},
            "orientation_module": {
                "path": str(module_root / "orientation.py"),
                "sha256": sha256_file(module_root / "orientation.py"),
            },
        },
    }


def _validate_rendered_case(case_dir: Path) -> None:
    executable = require_executable("foamDictionary")
    dictionaries = [
        *sorted((case_dir / "0").iterdir()),
        *sorted((case_dir / "constant").glob("*Properties")),
        *sorted((case_dir / "system").iterdir()),
    ]
    for dictionary in dictionaries:
        if not dictionary.is_file() or dictionary.name == "case-context.json":
            continue
        relative = dictionary.relative_to(case_dir).as_posix()
        entry = {"system/forces": "forces", "system/meshQualityDict": "maxNonOrtho"}.get(relative, "FoamFile")
        run_checked(
            [executable, dictionary, "-entry", entry],
            case_dir,
            case_dir / f"log.foamDictionary.{relative.replace('/', '-')}",
        )


class PipelineOrchestrator:
    def __init__(self, project_root: Path, config: PipelineConfig, runtime: OpenFoamRuntime) -> None:
        self.project_root = project_root.resolve(strict=True)
        self.config = config
        self.runtime = runtime

    def _accepted_mesh_context(
        self, input_path: Path
    ) -> tuple[RunPaths, dict[str, object], GeometryResult, dict[MeshProfile, MeshResult]]:
        paths, manifest, geometry, _ = _validated_mesh_run(
            self.project_root,
            input_path,
            self.config,
            MeshProfile.COARSE,
            self.runtime,
        )
        configuration = manifest.get("configuration")
        if not isinstance(configuration, dict):
            raise ValueError(f"run manifest has invalid configuration metadata: {paths.manifest}")
        if configuration.get("values") != self.config.raw:
            configuration["values"] = self.config.raw
            atomic_write_json(paths.manifest, manifest)
        discover_status = StageStatus(read_status(paths)["stages"][Stage.DISCOVER.value]["status"])
        if discover_status == StageStatus.PENDING:
            transition_stage(paths, Stage.DISCOVER, StageStatus.RUNNING, detail={"input": str(input_path)})
            transition_stage(
                paths,
                Stage.DISCOVER,
                StageStatus.ACCEPTED,
                detail={"source_sha256": sha256_file(input_path)},
            )
        return paths, manifest, geometry, _mesh_family(paths, manifest)

    def _build_mesh_context(
        self,
        input_path: Path,
        options: AnalysisOptions,
    ) -> tuple[RunPaths, dict[str, object], GeometryResult, dict[MeshProfile, MeshResult]]:
        source_digest = sha256_file(input_path)
        paths = create_run_paths(self.project_root, source_digest, self.config.digest)
        manifest: dict[str, object] = {
            "schema_version": 2,
            "mode": "full",
            "run_root": str(paths.root),
            "configuration": {
                "path": str(self.config.source),
                "digest": self.config.digest,
                "values": self.config.raw,
            },
            "runtime": {
                "version": self.runtime.version,
                "build": self.runtime.build,
                "project_dir": str(self.runtime.project_dir),
                "mpi_ranks": self.config.raw["mpi_ranks"],
            },
            "source": {"path": str(input_path), "sha256": source_digest},
            "mesh_revision": 0,
        }
        atomic_write_json(paths.manifest, manifest)
        transition_stage(paths, Stage.DISCOVER, StageStatus.RUNNING, detail={"input": str(input_path)})
        transition_stage(
            paths,
            Stage.DISCOVER,
            StageStatus.ACCEPTED,
            detail={"source_sha256": source_digest},
        )
        transition_stage(paths, Stage.GEOMETRY, StageStatus.RUNNING)
        try:
            geometry = prepare_geometry(input_path, paths, self.config, self.runtime, options.bow_override)
            manifest["geometry"] = _geometry_payload(geometry)
            atomic_write_json(paths.manifest, manifest)
            transition_stage(
                paths,
                Stage.GEOMETRY,
                StageStatus.ACCEPTED,
                detail={"source_sha256": source_digest, "geometry": str(paths.geometry)},
            )
        except (OSError, RuntimeError, ValueError):
            transition_stage(paths, Stage.GEOMETRY, StageStatus.REJECTED)
            raise
        transition_stage(paths, Stage.PREPARE, StageStatus.RUNNING)
        try:
            prepared: dict[MeshProfile, Path] = {}
            for profile in MeshProfile:
                case_dir = render_mesh_case(
                    paths,
                    geometry,
                    calculate_mesh_sizing(geometry, self.config, profile),
                    self.config,
                )
                _validate_rendered_case(case_dir)
                prepared[profile] = case_dir
            prepare_checkpoint = write_task_checkpoint(paths)
            manifest["case_rendering_digest"] = case_rendering_digest()
            manifest["prepared_cases"] = {
                profile.value: {
                    "path": str(case_dir),
                    "context_sha256": sha256_file(case_dir / "case-context.json"),
                    "snappy_hex_mesh_dict_sha256": sha256_file(case_dir / "system" / "snappyHexMeshDict"),
                }
                for profile, case_dir in prepared.items()
            }
            atomic_write_json(paths.manifest, manifest)
            transition_stage(
                paths,
                Stage.PREPARE,
                StageStatus.ACCEPTED,
                detail={"checkpoint": str(prepare_checkpoint)},
            )
        except (OSError, RuntimeError, ValueError):
            transition_stage(paths, Stage.PREPARE, StageStatus.REJECTED)
            raise
        transition_stage(paths, Stage.MESH, StageStatus.RUNNING)
        mesh_results: dict[MeshProfile, MeshResult] = {}
        try:
            for profile in MeshProfile:
                projection = _resource_projection(paths, profile)
                result = generate_mesh(prepared[profile], profile, self.runtime, self.config)
                result_path = prepared[profile] / "mesh-result.json"
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                payload["resource_projection"] = projection
                atomic_write_json(result_path, payload)
                mesh_results[profile] = result
                entries = manifest.setdefault("mesh_results", {})
                if not isinstance(entries, dict):
                    raise ValueError(f"manifest mesh_results has invalid type: {paths.manifest}")
                entries[profile.value] = {
                    "path": str(result_path),
                    "sha256": sha256_file(result_path),
                    "mesh_sha256": result.mesh_sha256,
                }
                atomic_write_json(paths.manifest, manifest)
            mesh_checkpoint = _checkpoint(self.project_root, paths, mesh_results)
            transition_stage(
                paths,
                Stage.MESH,
                StageStatus.ACCEPTED,
                detail={"checkpoint": str(mesh_checkpoint)},
            )
        except (OSError, RuntimeError, ValueError):
            transition_stage(paths, Stage.MESH, StageStatus.REJECTED)
            raise
        return paths, manifest, geometry, mesh_results

    def report_only(self, run_id: str) -> Path:
        run_root = (self.project_root / "runs" / run_id).resolve(strict=True)
        paths = _paths(run_root)
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        configuration = manifest.get("configuration")
        runtime = manifest.get("runtime")
        if not isinstance(configuration, dict) or configuration.get("digest") != self.config.digest:
            raise ValueError(f"run configuration does not match {self.config.source}: {paths.manifest}")
        if not isinstance(runtime, dict) or runtime.get("build") != self.runtime.build:
            raise ValueError(f"run OpenFOAM build does not match the active runtime: {paths.manifest}")
        configuration["values"] = self.config.raw
        geometry = _geometry_result(paths.manifest)
        meshes = _mesh_family(paths, manifest)
        grid_path = paths.postprocessing / "grid-study.json"
        study = _read_grid_study(grid_path)
        solver_entries = manifest.get("solver_results")
        if not isinstance(solver_entries, dict):
            raise ValueError(f"run has no solver results: {paths.manifest}")
        cases: list[CaseResult] = []
        for entry in solver_entries.values():
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                continue
            result_path = Path(entry["path"]).resolve(strict=True)
            if sha256_file(result_path) != entry.get("sha256"):
                raise ValueError(f"case-result evidence hash changed: {result_path}")
            cases.append(case_result_from_json(result_path))
        cases = _unique_cases(cases)
        if not cases:
            raise ValueError(f"run has no readable CFD case results: {paths.manifest}")
        markdown, html = render_run_report(
            paths,
            manifest,
            geometry,
            list(meshes.values()),
            study,
            cases,
        )
        storage = self.config.raw["storage"]
        store = ResultStore((self.project_root / str(storage["database"])).resolve())
        manifest["reports"] = {
            "markdown": str(markdown),
            "html": str(html),
            "comparison": str((self.project_root / str(storage["comparison_html"])).resolve()),
        }
        atomic_write_json(paths.manifest, manifest)
        store.persist_run(manifest, geometry, list(meshes.values()), study, cases)
        store.set_report_path(paths.root.name, html)
        summary = (self.project_root / str(storage["summary_csv"])).resolve()
        comparison = (self.project_root / str(storage["comparison_html"])).resolve()
        store.export_summary_csv(summary)
        render_comparison_report(store, comparison)
        _task_07_checkpoint(self.project_root, paths, store, summary, comparison, markdown, html)
        acceptance = _write_acceptance(
            paths,
            manifest,
            geometry,
            list(meshes.values()),
            study,
            cases,
            store.database,
            summary,
            html,
        )
        manifest["acceptance"] = {"path": str(acceptance), "sha256": sha256_file(acceptance)}
        atomic_write_json(paths.manifest, manifest)
        append_run_event(paths, {"event": "report_regenerated", "report": str(html)})
        return html

    def _persist_and_report(
        self,
        paths: RunPaths,
        manifest: dict[str, object],
        geometry: GeometryResult,
        meshes: dict[MeshProfile, MeshResult],
        study: GridStudyResult,
        cases: Sequence[CaseResult],
    ) -> tuple[tuple[str, ...], Path | None]:
        storage = self.config.raw["storage"]
        database = (self.project_root / str(storage["database"])).resolve()
        summary_csv = (self.project_root / str(storage["summary_csv"])).resolve()
        comparison_html = (self.project_root / str(storage["comparison_html"])).resolve()
        store = ResultStore(database)
        persist_status = StageStatus(read_status(paths)["stages"][Stage.PERSIST.value]["status"])
        report_status = StageStatus(read_status(paths)["stages"][Stage.REPORT.value]["status"])
        reports = manifest.get("reports")
        if persist_status == StageStatus.ACCEPTED and report_status == StageStatus.ACCEPTED:
            append_run_event(paths, {"event": "run_deduplicated", "database": str(database)})
            report_path = None
            if isinstance(reports, dict) and isinstance(reports.get("html"), str):
                report_path = Path(reports["html"])
            return ("deduplicated",), report_path
        if persist_status == StageStatus.REJECTED or report_status == StageStatus.REJECTED:
            raise ValueError("persistence/report stage is rejected; inspect the retained stage evidence")
        try:
            if persist_status == StageStatus.PENDING:
                transition_stage(paths, Stage.PERSIST, StageStatus.RUNNING)
            store.persist_run(manifest, geometry, list(meshes.values()), study, cases)
            if persist_status != StageStatus.ACCEPTED:
                transition_stage(
                    paths,
                    Stage.PERSIST,
                    StageStatus.ACCEPTED,
                    detail={"database": str(database)},
                )
        except (OSError, RuntimeError, ValueError, sqlite3.Error):
            current = StageStatus(read_status(paths)["stages"][Stage.PERSIST.value]["status"])
            if current == StageStatus.RUNNING:
                transition_stage(paths, Stage.PERSIST, StageStatus.REJECTED)
            raise
        try:
            if report_status == StageStatus.PENDING:
                transition_stage(paths, Stage.REPORT, StageStatus.RUNNING)
            markdown_report, html_report = render_run_report(
                paths,
                manifest,
                geometry,
                list(meshes.values()),
                study,
                cases,
            )
            manifest["reports"] = {
                "markdown": str(markdown_report),
                "html": str(html_report),
                "comparison": str(comparison_html),
            }
            atomic_write_json(paths.manifest, manifest)
            store.set_report_path(paths.root.name, html_report)
            store.export_summary_csv(summary_csv)
            render_comparison_report(store, comparison_html)
            checkpoint = _task_07_checkpoint(
                self.project_root,
                paths,
                store,
                summary_csv,
                comparison_html,
                markdown_report,
                html_report,
            )
            acceptance = _write_acceptance(
                paths,
                manifest,
                geometry,
                list(meshes.values()),
                study,
                cases,
                database,
                summary_csv,
                html_report,
            )
            manifest["task_07_checkpoint"] = {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}
            manifest["acceptance"] = {"path": str(acceptance), "sha256": sha256_file(acceptance)}
            atomic_write_json(paths.manifest, manifest)
            if report_status != StageStatus.ACCEPTED:
                transition_stage(
                    paths,
                    Stage.REPORT,
                    StageStatus.ACCEPTED,
                    detail={"report": str(html_report), "comparison": str(comparison_html)},
                )
            return (), html_report
        except (OSError, RuntimeError, ValueError, sqlite3.Error):
            current = StageStatus(read_status(paths)["stages"][Stage.REPORT.value]["status"])
            if current == StageStatus.RUNNING:
                transition_stage(paths, Stage.REPORT, StageStatus.REJECTED)
            raise

    def _reference(
        self,
        paths: RunPaths,
        manifest: dict[str, object],
        geometry: GeometryResult,
        meshes: dict[MeshProfile, MeshResult],
        options: AnalysisOptions,
    ) -> tuple[GridStudyResult, dict[MeshProfile, CaseResult], tuple[str, ...]]:
        status = read_status(paths)
        current = StageStatus(status["stages"][Stage.REFERENCE.value]["status"])
        grid_path = paths.postprocessing / "grid-study.json"
        if current == StageStatus.ACCEPTED and not options.force:
            study = _read_grid_study(grid_path)
            cases: dict[MeshProfile, CaseResult] = {}
            for profile in MeshProfile:
                result = _reusable_case(manifest, meshes[profile], float(self.config.raw["operating"]["reference_speed_m_s"]))
                if result is None:
                    raise ValueError(f"accepted reference checkpoint is incomplete for {profile.value}")
                cases[profile] = result
            append_run_event(paths, {"event": "stage_checkpoint_reused", "stage": Stage.REFERENCE.value})
            return study, cases, ()
        if current == StageStatus.REJECTED:
            raise ValueError("reference stage is rejected; start a new analysis run after correcting the cause")
        if current == StageStatus.PENDING:
            transition_stage(paths, Stage.REFERENCE, StageStatus.RUNNING)
        speed = float(self.config.raw["operating"]["reference_speed_m_s"])
        cases = {}
        diagnostics: list[str] = []
        reused_profiles: list[str] = []
        try:
            for profile in MeshProfile:
                result, reused = _solve_or_reuse(
                    paths,
                    manifest,
                    geometry,
                    meshes[profile],
                    speed,
                    self.runtime,
                    self.config,
                    force=options.force,
                )
                cases[profile] = result
                diagnostics.extend(result.diagnostic_codes)
                if reused:
                    reused_profiles.append(profile.value)
            study = evaluate_grid_study(
                cases[MeshProfile.FINE],
                cases[MeshProfile.MEDIUM],
                cases[MeshProfile.COARSE],
                meshes[MeshProfile.FINE],
                meshes[MeshProfile.MEDIUM],
                meshes[MeshProfile.COARSE],
                self.config,
            )
            payload = _grid_study_payload(paths, study, cases)
            atomic_write_json(grid_path, payload)
            manifest["grid_study"] = {
                "path": str(grid_path),
                "sha256": sha256_file(grid_path),
                "accepted": study.accepted,
                "classification": study.classification,
                "qualification_scope": "reference_speed_only",
            }
            manifest.setdefault("resume", {})
            if isinstance(manifest["resume"], dict):
                manifest["resume"]["reference_reused_profiles"] = reused_profiles
            atomic_write_json(paths.manifest, manifest)
            if not study.accepted:
                transition_stage(
                    paths,
                    Stage.REFERENCE,
                    StageStatus.REJECTED,
                    detail={"reason": study.reason, "grid_study": str(grid_path)},
                )
                return study, cases, tuple(dict.fromkeys(diagnostics))
            transition_stage(
                paths,
                Stage.REFERENCE,
                StageStatus.ACCEPTED,
                detail={"grid_study": str(grid_path), "sha256": sha256_file(grid_path)},
            )
            return study, cases, tuple(dict.fromkeys(diagnostics))
        except (OSError, RuntimeError, ValueError):
            current = StageStatus(read_status(paths)["stages"][Stage.REFERENCE.value]["status"])
            if current == StageStatus.RUNNING:
                transition_stage(paths, Stage.REFERENCE, StageStatus.REJECTED)
            raise

    def _sweep(
        self,
        paths: RunPaths,
        manifest: dict[str, object],
        geometry: GeometryResult,
        medium_mesh: MeshResult,
        reference_cases: dict[MeshProfile, CaseResult],
        study: GridStudyResult,
        options: AnalysisOptions,
    ) -> tuple[list[CaseResult], tuple[str, ...]]:
        if not study.accepted:
            raise ValueError(f"production sweep requires accepted grid qualification: {study.reason}")
        current = StageStatus(read_status(paths)["stages"][Stage.SWEEP.value]["status"])
        if current == StageStatus.REJECTED:
            raise ValueError("speed-sweep stage is rejected; start a new analysis run after correcting the cause")
        if current == StageStatus.PENDING:
            transition_stage(paths, Stage.SWEEP, StageStatus.RUNNING)
        speeds = sorted(float(value) for value in self.config.raw["operating"]["speeds_m_s"])
        reference_speed = float(self.config.raw["operating"]["reference_speed_m_s"])
        results: list[CaseResult] = []
        diagnostics: list[str] = []
        try:
            for speed in speeds:
                if math.isclose(speed, reference_speed, rel_tol=0.0, abs_tol=1.0e-12) and not options.force:
                    result = reference_cases[MeshProfile.MEDIUM]
                    append_run_event(
                        paths,
                        {
                            "event": "case_checkpoint_reused",
                            "result_key": _solver_key(MeshProfile.MEDIUM, speed),
                            "reason": "accepted reference case",
                        },
                    )
                else:
                    result, _ = _solve_or_reuse(
                        paths,
                        manifest,
                        geometry,
                        medium_mesh,
                        speed,
                        self.runtime,
                        self.config,
                        force=options.force,
                    )
                results.append(result)
                diagnostics.extend(result.diagnostic_codes)
            if all(result.accepted for result in results):
                drag = [result.drag_n for result in results]
                if any(next_value <= value for value, next_value in zip(drag, drag[1:])):
                    diagnostics.append("warning_non_monotonic_drag")
                manifest["production_sweep"] = {
                    "qualification_scope": "reference_speed_only",
                    "mesh_profile": MeshProfile.MEDIUM.value,
                    "mesh_sha256": medium_mesh.mesh_sha256,
                    "case_keys": [_solver_key(MeshProfile.MEDIUM, speed) for speed in speeds],
                    "accepted": True,
                    "diagnostic_codes": list(dict.fromkeys(diagnostics)),
                }
                atomic_write_json(paths.manifest, manifest)
                if current != StageStatus.ACCEPTED:
                    transition_stage(
                        paths,
                        Stage.SWEEP,
                        StageStatus.ACCEPTED,
                        detail={"accepted_cases": len(results)},
                    )
            else:
                manifest["production_sweep"] = {
                    "qualification_scope": "reference_speed_only",
                    "mesh_profile": MeshProfile.MEDIUM.value,
                    "mesh_sha256": medium_mesh.mesh_sha256,
                    "case_keys": [_solver_key(MeshProfile.MEDIUM, speed) for speed in speeds],
                    "accepted": False,
                    "diagnostic_codes": list(dict.fromkeys(diagnostics)),
                }
                atomic_write_json(paths.manifest, manifest)
                transition_stage(
                    paths,
                    Stage.SWEEP,
                    StageStatus.REJECTED,
                    detail={"rejected_cases": sum(1 for result in results if not result.accepted)},
                )
            return results, tuple(dict.fromkeys(diagnostics))
        except (OSError, RuntimeError, ValueError):
            stage_status = StageStatus(read_status(paths)["stages"][Stage.SWEEP.value]["status"])
            if stage_status == StageStatus.RUNNING:
                transition_stage(paths, Stage.SWEEP, StageStatus.REJECTED)
            raise

    def run(self, input_path: Path, options: AnalysisOptions) -> RunOutcome:
        input_path = input_path.resolve(strict=True)
        if options.restart or options.force:
            paths, manifest, geometry, meshes = self._build_mesh_context(input_path, options)
        else:
            try:
                paths, manifest, geometry, meshes = self._accepted_mesh_context(input_path)
            except ValueError:
                paths, manifest, geometry, meshes = self._build_mesh_context(input_path, options)
        study, reference_cases, reference_diagnostics = self._reference(
            paths,
            manifest,
            geometry,
            meshes,
            options,
        )
        reference_results = list(reference_cases.values())
        study_payload = json.loads((paths.postprocessing / "grid-study.json").read_text(encoding="utf-8"))
        if not study.accepted:
            _task_06_checkpoint(
                self.project_root,
                paths,
                manifest,
                study_payload,
                (),
                reference_diagnostics,
            )
            return RunOutcome(
                run_id=paths.root.name,
                status=StageStatus.REJECTED,
                run_paths=paths,
                accepted_case_count=_accepted_count(reference_results),
                diagnostic_codes=reference_diagnostics,
            )
        if options.reference_only or options.stop_after == Stage.REFERENCE:
            _task_06_checkpoint(
                self.project_root,
                paths,
                manifest,
                study_payload,
                (reference_cases[MeshProfile.MEDIUM],),
                reference_diagnostics,
            )
            return RunOutcome(
                run_id=paths.root.name,
                status=StageStatus.ACCEPTED,
                run_paths=paths,
                accepted_case_count=_accepted_count(reference_results),
                diagnostic_codes=reference_diagnostics,
            )
        production, sweep_diagnostics = self._sweep(
            paths,
            manifest,
            geometry,
            meshes[MeshProfile.MEDIUM],
            reference_cases,
            study,
            options,
        )
        diagnostics = tuple(dict.fromkeys((*reference_diagnostics, *sweep_diagnostics)))
        checkpoint = _task_06_checkpoint(
            self.project_root,
            paths,
            manifest,
            study_payload,
            production,
            diagnostics,
        )
        append_run_event(paths, {"event": "task_checkpoint_written", "task": 6, "path": str(checkpoint)})
        accepted = all(result.accepted for result in production)
        report_diagnostics: tuple[str, ...] = ()
        if accepted:
            all_cases = _unique_cases([*reference_results, *production])
            report_diagnostics, _ = self._persist_and_report(
                paths,
                manifest,
                geometry,
                meshes,
                study,
                all_cases,
            )
            diagnostics = tuple(dict.fromkeys((*diagnostics, *report_diagnostics)))
        return RunOutcome(
            run_id=paths.root.name,
            status=StageStatus.ACCEPTED if accepted else StageStatus.REJECTED,
            run_paths=paths,
            accepted_case_count=_accepted_count([*reference_results, *production]),
            diagnostic_codes=diagnostics,
        )
