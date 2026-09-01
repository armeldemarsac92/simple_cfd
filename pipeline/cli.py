from __future__ import annotations

import argparse
import json
import importlib.metadata
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from pipeline import __version__
from pipeline.config import load_config
from pipeline.cad import prepare_geometry
from pipeline.case_builder import case_rendering_digest, calculate_mesh_sizing, render_mesh_case, write_task_checkpoint
from pipeline.models import AnalysisOptions, GeometryResult, MeshProfile, RunPaths, Stage, StageStatus
from pipeline.orchestrator import PipelineOrchestrator, run_mesh_only, run_solve_one
from pipeline.runtime import discover_openfoam
from pipeline.state import atomic_write_json, create_run_paths, sha256_file, transition_stage


_DEPENDENCIES = (
    ("gmsh", "gmsh"),
    ("Jinja2", "jinja2"),
    ("matplotlib", "matplotlib"),
    ("meshio", "meshio"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("shapely", "shapely"),
    ("trimesh", "trimesh"),
)
_GIB = 1024**3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analyze_hull.sh")
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--mesh-only", choices=["coarse", "medium", "fine"])
    parser.add_argument("--solve-one", nargs=2, metavar=("SPEED", "PROFILE"))
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--report-only", metavar="RUN_ID")
    parser.add_argument("--stop-after", choices=[stage.value for stage in Stage])
    parser.add_argument("--bow", choices=["+x", "-x", "+y", "-y", "+z", "-z"])
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _physical_memory_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return 0


def _doctor(config_path: Path) -> int:
    config = load_config(config_path)
    runtime = discover_openfoam()
    versions = []
    for distribution, module in _DEPENDENCIES:
        __import__(module)
        versions.append((distribution, importlib.metadata.version(distribution)))
    disk = shutil.disk_usage(config.source.parent.parent)
    memory_bytes = _physical_memory_bytes()
    print(f"Pipeline version: {__version__}")
    print(f"Python: {sys.version.split()[0]}")
    for distribution, version in versions:
        print(f"{distribution}: {version}")
    print(f"OpenFOAM: {runtime.version} build {runtime.build}")
    print(f"pimpleFoam: {runtime.pimple_foam}")
    print(f"mpirun: {runtime.mpirun}")
    print(f"Selected MPI ranks: {config.raw['mpi_ranks']}")
    print(f"Configuration digest: {config.digest}")
    print(f"Free disk: {disk.free / _GIB:.2f} GiB")
    print(f"Physical RAM: {memory_bytes / _GIB:.2f} GiB")
    if disk.free < 30 * _GIB:
        raise RuntimeError("at least 30 GiB of free disk is required for the CFD pipeline")
    if memory_bytes < 30 * _GIB:
        raise RuntimeError("at least 30 GiB of physical RAM is required for the CFD pipeline")
    return 0


def _execution_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    selected = [
        ("doctor", args.doctor),
        ("geometry-only", args.geometry_only),
        ("prepare-only", args.prepare_only),
        ("mesh-only", args.mesh_only is not None),
        ("solve-one", args.solve_one is not None),
        ("reference-only", args.reference_only),
        ("report-only", args.report_only is not None),
    ]
    modes = [name for name, enabled in selected if enabled]
    if len(modes) > 1:
        parser.error(f"mutually incompatible execution modes: {', '.join(modes)}")
    mode = modes[0] if modes else "full"
    if mode == "doctor" and (args.input is not None or args.stop_after or args.bow or args.restart or args.force):
        parser.error("--doctor cannot be combined with analysis inputs or execution modifiers")
    if mode == "report-only" and (args.input is not None or args.stop_after or args.bow):
        parser.error("--report-only cannot be combined with an input, --bow, or --stop-after")
    return mode


def _discover_inputs(input_path: Path | None, project_root: Path) -> list[Path]:
    if input_path is not None:
        resolved = input_path.resolve()
        if not resolved.is_file():
            raise ValueError(f"input hull does not exist: {input_path}")
        if resolved.suffix.lower() not in {".step", ".stp"}:
            raise ValueError(f"input hull must use .step or .stp: {input_path}")
        return [resolved]
    inbox = project_root / "hulls" / "inbox"
    return sorted([*inbox.glob("*.step"), *inbox.glob("*.stp")], key=lambda path: path.name)


def _geometry_manifest(result: GeometryResult) -> dict[str, object]:
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
            "orientation_metadata": {"path": str(result.orientation_metadata), "sha256": sha256_file(result.orientation_metadata)},
            "preview_png": {"path": str(result.preview_png), "sha256": sha256_file(result.preview_png)},
            "surface_check_log": {"path": str(result.surface_check_log), "sha256": sha256_file(result.surface_check_log)},
            "cad_module": {"path": "pipeline/cad.py", "sha256": sha256_file(Path("pipeline/cad.py"))},
            "orientation_module": {"path": "pipeline/orientation.py", "sha256": sha256_file(Path("pipeline/orientation.py"))},
        },
    }


def _geometry_only(input_path: Path, project_root: Path, config_path: Path, bow_override: str | None) -> GeometryResult:
    config = load_config(config_path)
    runtime = discover_openfoam()
    source_sha256 = sha256_file(input_path)
    paths = create_run_paths(project_root, source_sha256, config.digest)
    manifest: dict[str, object] = {
        "schema_version": 2,
        "run_root": str(paths.root),
        "mode": "geometry-only",
        "configuration": {"path": str(config.source), "digest": config.digest},
        "runtime": {"version": runtime.version, "build": runtime.build, "project_dir": str(runtime.project_dir)},
        "source": {"path": str(input_path), "sha256": source_sha256},
    }
    atomic_write_json(paths.manifest, manifest)
    transition_stage(paths, Stage.GEOMETRY, StageStatus.RUNNING)
    try:
        result = prepare_geometry(input_path, paths, config, runtime, bow_override)
    except (OSError, RuntimeError, ValueError):
        transition_stage(paths, Stage.GEOMETRY, StageStatus.REJECTED)
        raise
    manifest["geometry"] = _geometry_manifest(result)
    atomic_write_json(paths.manifest, manifest)
    transition_stage(paths, Stage.GEOMETRY, StageStatus.ACCEPTED, detail={"manifest": str(paths.manifest)})
    evidence_path = project_root / "docs" / "superpowers" / "plans" / "evidence" / "task-02-geometry.json"
    atomic_write_json(evidence_path, json.loads(paths.manifest.read_text(encoding="utf-8")))
    print(f"Geometry accepted: {paths.root}")
    print(f"Length: {result.length_m:.6f} m; frontal area: {result.frontal_area_m2:.6f} m2")
    print(f"Bow: {result.bow_original_axis}; confidence: {result.bow_confidence:.3f}")
    return result


def _triple(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"accepted geometry manifest has invalid {name}")
    return tuple(float(component) for component in value)


def _matrix(value: object, name: str) -> tuple[tuple[float, float, float, float], ...]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"accepted geometry manifest has invalid {name}")
    rows = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"accepted geometry manifest has invalid {name}")
        rows.append(tuple(float(component) for component in row))
    return tuple(rows)


def _geometry_from_manifest(manifest_path: Path) -> GeometryResult:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    geometry = manifest.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError(f"accepted geometry manifest lacks geometry data: {manifest_path}")
    artifacts = geometry.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"accepted geometry manifest lacks artifact data: {manifest_path}")

    def artifact(name: str) -> Path:
        entry = artifacts.get(name)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"accepted geometry manifest lacks {name}: {manifest_path}")
        path = Path(entry["path"])
        if not path.is_file():
            raise ValueError(f"accepted geometry artifact is unavailable: {path}")
        return path.resolve()

    original_bounds = geometry.get("original_bounds_m")
    if not isinstance(original_bounds, list) or len(original_bounds) != 2:
        raise ValueError(f"accepted geometry manifest has invalid original bounds: {manifest_path}")
    source = Path(str(geometry["source"])).resolve()
    if not source.is_file():
        raise ValueError(f"accepted geometry source is unavailable: {source}")
    return GeometryResult(
        source=source,
        source_sha256=str(geometry["source_sha256"]),
        declared_unit=str(geometry["declared_unit"]),
        scale_to_m=float(geometry["scale_to_m"]),
        normalized_stl=artifact("normalized_stl"),
        original_bounds_m=(_triple(original_bounds[0], "original_bounds_m"), _triple(original_bounds[1], "original_bounds_m")),
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


def _accepted_geometry_run(project_root: Path, input_path: Path, config_digest: str) -> tuple[GeometryResult, object]:
    source_sha256 = sha256_file(input_path)
    manifests = sorted((project_root / "runs").glob("*/manifest.json"), reverse=True)
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source = manifest.get("source")
            configuration = manifest.get("configuration")
            if not isinstance(source, dict) or not isinstance(configuration, dict):
                continue
            if source.get("sha256") != source_sha256 or configuration.get("digest") != config_digest:
                continue
            result = _geometry_from_manifest(manifest_path)
            root = manifest_path.parent.resolve()
            paths = RunPaths(
                root=root,
                manifest=manifest_path.resolve(),
                status=root / "status.json",
                geometry=root / "geometry",
                meshes=root / "meshes",
                cases=root / "cases",
                postprocessing=root / "postprocessing",
            )
            return result, paths
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    raise ValueError(
        "--prepare-only requires an accepted Task 2 geometry checkpoint for this input and configuration; "
        "run --geometry-only first"
    )


def _parse_rendered_dictionaries(case_dir: Path) -> None:
    executable = shutil.which("foamDictionary")
    if executable is None:
        raise RuntimeError("foamDictionary is unavailable after sourcing OpenFOAM v2512")
    dictionaries = [
        *sorted((case_dir / "0").iterdir()),
        *sorted((case_dir / "constant").glob("*Properties")),
        *sorted((case_dir / "system").iterdir()),
    ]
    for dictionary in dictionaries:
        if not dictionary.is_file() or dictionary.name == "case-context.json":
            continue
        safe_name = dictionary.relative_to(case_dir).as_posix().replace("/", "-")
        entry = {
            "system/forces": "forces",
            "system/meshQualityDict": "maxNonOrtho",
        }.get(dictionary.relative_to(case_dir).as_posix(), "FoamFile")
        command = [executable, str(dictionary), "-entry", entry]
        completed = subprocess.run(command, cwd=case_dir, text=True, capture_output=True)
        log_path = case_dir / f"log.foamDictionary.{safe_name}"
        log_path.write_text(
            f"cwd={case_dir}\nargv={command!r}\nexit_status={completed.returncode}\n\n"
            f"{completed.stdout}{completed.stderr}",
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(f"foamDictionary failed for {dictionary}; log: {log_path}")


def _prepare_only(input_path: Path, project_root: Path, config_path: Path) -> list[Path]:
    config = load_config(config_path)
    discover_openfoam()
    geometry, paths = _accepted_geometry_run(project_root, input_path, config.digest)
    if any(paths.cases.iterdir()):
        previous_manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        reused_from = paths.root
        paths = create_run_paths(project_root, geometry.source_sha256, config.digest)
        manifest = {
            "schema_version": 2,
            "mode": "prepare-only",
            "run_root": str(paths.root),
            "geometry_reused_from": str(reused_from),
            "configuration": {"path": str(config.source), "digest": config.digest},
            "source": {"path": str(geometry.source), "sha256": geometry.source_sha256},
            "geometry": previous_manifest["geometry"],
        }
        atomic_write_json(paths.manifest, manifest)
        transition_stage(paths, Stage.GEOMETRY, StageStatus.RUNNING)
        transition_stage(paths, Stage.GEOMETRY, StageStatus.ACCEPTED, detail={"reused_from": str(reused_from)})
    transition_stage(paths, Stage.PREPARE, StageStatus.RUNNING)
    try:
        prepared = []
        for profile in MeshProfile:
            sizing = calculate_mesh_sizing(geometry, config, profile)
            case_dir = render_mesh_case(paths, geometry, sizing, config)
            _parse_rendered_dictionaries(case_dir)
            prepared.append(case_dir)
        checkpoint = write_task_checkpoint(paths)
    except (OSError, RuntimeError, ValueError):
        transition_stage(paths, Stage.PREPARE, StageStatus.REJECTED)
        raise
    transition_stage(
        paths,
        Stage.PREPARE,
        StageStatus.ACCEPTED,
        detail={"cases": [str(path) for path in prepared], "checkpoint": str(checkpoint)},
    )
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest["case_rendering_digest"] = case_rendering_digest()
    manifest["prepared_cases"] = {
        profile.value: {
            "path": str(case_dir),
            "context_sha256": sha256_file(case_dir / "case-context.json"),
            "snappy_hex_mesh_dict_sha256": sha256_file(case_dir / "system" / "snappyHexMeshDict"),
        }
        for profile, case_dir in zip(MeshProfile, prepared, strict=True)
    }
    atomic_write_json(paths.manifest, manifest)
    return prepared


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        mode = _execution_mode(args, parser)
        if mode == "doctor":
            return _doctor(args.config)
        config = load_config(args.config)
        project_root = config.source.parent.parent
        if mode == "report-only":
            runtime = discover_openfoam()
            report = PipelineOrchestrator(project_root, config, runtime).report_only(str(args.report_only))
            print(f"Report regenerated: {report}")
            return 0
        inputs = _discover_inputs(args.input, project_root)
        if not inputs:
            raise ValueError(f"no .step or .stp hulls found in {project_root / 'hulls' / 'inbox'}")
        requested = ", ".join(str(path) for path in inputs)
        print(f"Selected hull input(s): {requested}")
        print(f"Requested mode: {mode}")
        if mode == "geometry-only":
            for input_path in inputs:
                _geometry_only(input_path, project_root, args.config, args.bow)
            return 0
        if mode == "prepare-only":
            for input_path in inputs:
                for case_dir in _prepare_only(input_path, project_root, args.config):
                    print(f"Prepared {case_dir.name} case: {case_dir}")
            return 0
        if mode == "mesh-only":
            runtime = discover_openfoam()
            for input_path in inputs:
                result, checkpoint = run_mesh_only(
                    input_path,
                    MeshProfile(args.mesh_only),
                    project_root,
                    config,
                    runtime,
                )
                print(
                    f"Mesh accepted: {result.profile.value}; cells={result.cell_count}; "
                    f"maxNonOrtho={result.max_non_orthogonality:.6g}; maxSkewness={result.max_skewness:.6g}; "
                    f"layers={result.layer_coverage_fraction:.3%}"
                )
                if checkpoint is not None:
                    print(f"Mesh family checkpoint: {checkpoint}")
            return 0
        if mode == "solve-one":
            runtime = discover_openfoam()
            speed_text, profile_text = args.solve_one
            try:
                speed = float(speed_text)
                profile = MeshProfile(profile_text)
            except ValueError as error:
                raise ValueError("--solve-one requires a numeric configured speed and coarse, medium, or fine") from error
            accepted = True
            for input_path in inputs:
                result, checkpoint = run_solve_one(
                    input_path,
                    speed,
                    profile,
                    project_root,
                    config,
                    runtime,
                )
                accepted = accepted and result.accepted
                print(
                    f"Case {'accepted' if result.accepted else 'rejected'}: {result.mesh_profile.value} "
                    f"at {result.speed_m_s:.2f} m/s; drag={result.drag_n:.6g} N; "
                    f"Cd={result.drag_coefficient:.6g}; yPlus median={result.yplus['median']:.6g}"
                )
                if result.diagnostic_codes:
                    print(f"Diagnostics: {', '.join(result.diagnostic_codes)}")
                if checkpoint is not None:
                    print(f"Task 5 case checkpoint: {checkpoint}")
            return 0 if accepted else 2
        if mode in {"reference-only", "full"}:
            runtime = discover_openfoam()
            accepted = True
            for input_path in inputs:
                orchestrator = PipelineOrchestrator(project_root, config, runtime)
                outcome = orchestrator.run(
                    input_path,
                    AnalysisOptions(
                        bow_override=args.bow,
                        stop_after=None if args.stop_after is None else Stage(args.stop_after),
                        restart=args.restart,
                        force=args.force,
                        reference_only=mode == "reference-only",
                    ),
                )
                grid_path = outcome.run_paths.postprocessing / "grid-study.json"
                grid_payload = json.loads(grid_path.read_text(encoding="utf-8"))["study"]
                print(
                    f"Grid qualification {'accepted' if grid_payload['accepted'] else 'rejected'}: "
                    f"{grid_payload['classification']}; p={grid_payload['observed_order']}; "
                    f"fine GCI={grid_payload['fine_gci_percent']}%"
                )
                print(
                    f"Run {outcome.run_id}: {outcome.status.value}; "
                    f"accepted CFD cases={outcome.accepted_case_count}"
                )
                if outcome.diagnostic_codes:
                    print(f"Diagnostics: {', '.join(outcome.diagnostic_codes)}")
                accepted = accepted and outcome.status == StageStatus.ACCEPTED
            return 0 if accepted else 2
        print("This execution mode is not available.")
        return 2
    except (OSError, RuntimeError, ValueError, sqlite3.Error, importlib.metadata.PackageNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
