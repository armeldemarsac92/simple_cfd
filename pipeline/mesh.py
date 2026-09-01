from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import trimesh

from pipeline.models import MeshProfile, MeshResult, OpenFoamRuntime, PipelineConfig
from pipeline.runtime import CommandFailure, require_executable
from pipeline.state import atomic_write_json, sha256_file


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_COMMANDS = (
    ("blockMesh", ("blockMesh",)),
    ("surfaceFeatureExtract", ("surfaceFeatureExtract",)),
    ("decomposePar", ("decomposePar", "-force")),
    (
        "snappyHexMesh",
        ("mpirun", "--bind-to", "core", "--map-by", "core", "-np", "{ranks}", "snappyHexMesh", "-parallel", "-overwrite"),
    ),
    ("reconstructParMesh", ("reconstructParMesh", "-constant")),
    (
        "checkMeshAllGeometry",
        ("checkMesh", "-constant", "-allGeometry", "-allTopology", "-meshQuality"),
    ),
    ("checkMesh", ("checkMesh", "-constant", "-meshQuality")),
)
_SURFACE_AUDIT_LINES = (
    "Surface has no illegal triangles.",
    "Surface is closed. All edges connected to two faces.",
    "Number of unconnected parts : 1",
    "Number of zones (connected area with consistent normal) : 1",
    "Surface is not self-intersecting",
)


def _as_json(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, MeshProfile):
        return value.value
    if isinstance(value, dict):
        return {str(key): _as_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_as_json(item) for item in value]
    return value


def _exit_status(text: str) -> int:
    matches = re.findall(r"^exit_status=(\d+)\s*$", text, flags=re.MULTILINE)
    if not matches:
        raise ValueError("mesh log lacks an exit_status footer")
    return int(matches[-1])


def _required_match(text: str, patterns: Sequence[str], name: str, cast: type[int] | type[float]) -> int | float:
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return cast(matches[-1])
    raise ValueError(f"checkMesh log lacks required metric: {name}")


def _mesh_digest(case_dir: Path) -> str:
    poly_mesh = case_dir / "constant" / "polyMesh"
    files = sorted(path for path in poly_mesh.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"reconstructed polyMesh is unavailable: {poly_mesh}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(poly_mesh).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _parse_boundary_face_count(case_dir: Path, patch: str = "hull") -> int:
    boundary = case_dir / "constant" / "polyMesh" / "boundary"
    text = boundary.read_text(encoding="utf-8", errors="replace")
    match = re.search(rf"\b{re.escape(patch)}\s*\{{(?P<body>.*?)\}}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"reconstructed boundary lacks patch {patch!r}: {boundary}")
    count = re.search(r"\bnFaces\s+(\d+)\s*;", match.group("body"))
    if count is None:
        raise ValueError(f"reconstructed boundary patch {patch!r} lacks nFaces: {boundary}")
    return int(count.group(1))


def _processor_hull_layer_counts(case_dir: Path) -> list[int]:
    counts: list[int] = []
    processor_dirs = sorted(path for path in case_dir.glob("processor*") if path.is_dir())
    if not processor_dirs:
        raise ValueError("snappyHexMesh did not preserve processor nSurfaceLayers fields for layer audit")
    for processor_dir in processor_dirs:
        field_paths = sorted(path for path in processor_dir.glob("*/nSurfaceLayers") if path.is_file())
        if len(field_paths) != 1:
            raise ValueError(
                f"expected exactly one processor nSurfaceLayers field, found {len(field_paths)}: {processor_dir}"
            )
        field_path = field_paths[0]
        raw = field_path.read_bytes()
        # latin-1 keeps byte offsets stable for OpenFOAM binary scalar lists
        # while leaving the surrounding dictionary syntax searchable.
        text = raw.decode("latin-1")
        patch = re.search(r"^\s*hull\s*\{(?P<body>.*?)^\s*\}", text, flags=re.DOTALL | re.MULTILINE)
        if patch is None:
            raise ValueError(f"nSurfaceLayers field lacks hull boundary data: {field_path}")
        local_face_count = _parse_boundary_face_count(processor_dir)
        values = re.search(r"\bvalue\s+nonuniform\s+List<scalar>\s+(\d+)\s*\(", patch.group("body"))
        raw_values: list[float]
        if values is not None:
            declared_count = int(values.group(1))
            absolute_start = patch.start("body") + values.end()
            if re.search(r"\bformat\s+binary\s*;", text[: patch.start()]) is not None:
                arch = re.search(r'\barch\s+"(LSB|MSB);label=\d+;scalar=(32|64)"\s*;', text[: patch.start()])
                if arch is None:
                    raise ValueError(f"binary nSurfaceLayers field lacks a supported arch header: {field_path}")
                cursor = absolute_start
                while cursor < len(raw) and raw[cursor] in b" \t\r\n":
                    cursor += 1
                scalar_bytes = int(arch.group(2)) // 8
                payload_end = cursor + declared_count * scalar_bytes
                if payload_end > len(raw):
                    raise ValueError(f"binary nSurfaceLayers hull payload is truncated: {field_path}")
                code = "f" if scalar_bytes == 4 else "d"
                byte_order = "<" if arch.group(1) == "LSB" else ">"
                raw_values = list(struct.unpack(f"{byte_order}{declared_count}{code}", raw[cursor:payload_end]))
                trailer = payload_end
                while trailer < len(raw) and raw[trailer] in b" \t\r\n":
                    trailer += 1
                if trailer >= len(raw) or raw[trailer] != ord(")"):
                    raise ValueError(f"binary nSurfaceLayers hull payload has an invalid terminator: {field_path}")
            else:
                list_end = patch.group("body").find(")", values.end())
                if list_end < 0:
                    raise ValueError(f"ASCII nSurfaceLayers hull list is unterminated: {field_path}")
                raw_values = [float(value) for value in re.findall(_FLOAT, patch.group("body")[values.end() : list_end])]
            if len(raw_values) != declared_count:
                raise ValueError(f"nSurfaceLayers hull boundary count mismatch: {field_path}")
            if len(raw_values) != local_face_count:
                raise ValueError(f"nSurfaceLayers hull boundary does not match processor patch: {field_path}")
        else:
            uniform = re.search(rf"\bvalue\s+uniform\s+({_FLOAT})\s*;", patch.group("body"))
            if uniform is None and local_face_count == 0:
                # OpenFOAM omits `value` for a calculated boundary patch that
                # has no local faces on this processor.
                raw_values = []
            elif uniform is None:
                raise ValueError(f"nSurfaceLayers hull boundary is not a scalar field: {field_path}")
            else:
                raw_values = [float(uniform.group(1))] * local_face_count
        parsed: list[int] = []
        for raw in raw_values:
            value = float(raw)
            rounded = round(value)
            if not math.isfinite(value) or value < 0 or not math.isclose(value, rounded, abs_tol=1.0e-9):
                raise ValueError(f"nSurfaceLayers contains an invalid layer count {raw!r}: {field_path}")
            parsed.append(int(rounded))
        counts.extend(parsed)
    return counts


def _parse_layer_metrics(case_dir: Path, log_path: Path, surface_face_count: int) -> tuple[int, float, int, int]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    extrusions = re.findall(r"Extruding\s+(\d+)\s+out of\s+(\d+)\s+faces\b", text, flags=re.IGNORECASE)
    if not extrusions:
        raise ValueError("snappyHexMesh log lacks required final hull extrusion statistics")
    _, logged_total = (int(value) for value in extrusions[-1])
    if logged_total != surface_face_count:
        raise ValueError(
            f"snappyHexMesh hull extrusion total {logged_total} does not match reconstructed hull faces {surface_face_count}"
        )
    layer_counts = _processor_hull_layer_counts(case_dir)
    if len(layer_counts) != surface_face_count:
        raise ValueError(
            f"nSurfaceLayers hull boundary total {len(layer_counts)} does not match reconstructed hull faces {surface_face_count}"
        )
    faces_with_layers = sum(value > 0 for value in layer_counts)
    return (
        faces_with_layers,
        faces_with_layers / surface_face_count,
        min(layer_counts),
        max(layer_counts),
    )


def _illegal_face_count(text: str, mesh_ok: bool) -> int:
    explicit = re.findall(rf"(?:illegal faces|illegalFaces)\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
    if explicit:
        return int(explicit[-1])
    face_tet_errors = re.findall(r"Error in face tets:\s*(\d+)\s+faces?\b", text, flags=re.IGNORECASE)
    if face_tet_errors:
        return int(face_tet_errors[-1])
    failed_sets = re.findall(r"Writing\s+(\d+)\s+(?:illegal|bad)\s+faces?\b", text, flags=re.IGNORECASE)
    if failed_sets:
        return sum(int(value) for value in failed_sets)
    face_proofs = (
        "Face tets OK.",
        "Face pyramids OK.",
        "Face-face connectivity OK.",
    )
    if all(proof in text for proof in face_proofs):
        return 0
    raise ValueError("checkMesh log lacks explicit illegal-face evidence")


def _concave_cell_count(text: str) -> int:
    matches = re.findall(
        r"Concave cells \(using face planes\) found,\s*number of cells:\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        return int(matches[-1])
    if "Concave cell check OK." in text:
        return 0
    raise ValueError("allGeometry checkMesh log lacks concave-cell evidence")


def parse_check_mesh(log: Path | str, all_geometry_log: Path | str | None = None) -> MeshResult:
    log_path = Path(log).resolve()
    text = log_path.read_text(encoding="utf-8", errors="replace")
    exit_code = _exit_status(text)
    mesh_ok = bool(re.search(r"^Mesh OK\.\s*$", text, flags=re.MULTILINE))
    diagnostics_path = log_path if all_geometry_log is None else Path(all_geometry_log).resolve()
    diagnostics_text = diagnostics_path.read_text(encoding="utf-8", errors="replace")
    diagnostics_exit_code = _exit_status(diagnostics_text)
    diagnostics_mesh_ok = bool(re.search(r"^Mesh OK\.\s*$", diagnostics_text, flags=re.MULTILINE))
    cell_count = int(_required_match(text, (r"^\s*cells:\s*(\d+)\s*$",), "cell_count", int))
    max_non_orthogonality = float(
        _required_match(
            text,
            (rf"Mesh non-orthogonality Max:\s*({_FLOAT})", rf"max(?:imum)? non-orthogonality\s*[=:]\s*({_FLOAT})"),
            "max_non_orthogonality",
            float,
        )
    )
    max_skewness = float(
        _required_match(
            text,
            (rf"Max skewness\s*=\s*({_FLOAT})", rf"maximum skewness\s*[=:]\s*({_FLOAT})"),
            "max_skewness",
            float,
        )
    )
    min_volume = float(
        _required_match(text, (rf"Min volume\s*=\s*({_FLOAT})",), "min_volume", float)
    )
    diagnostic_min_wellposedness_determinant = float(
        _required_match(
            diagnostics_text,
            (rf"Min determinant\s*=\s*({_FLOAT})", rf"Cell determinant .*?minimum:\s*({_FLOAT})"),
            "diagnostic_min_wellposedness_determinant",
            float,
        )
    )
    disconnected_regions = int(
        _required_match(
            text,
            (r"Number of regions:\s*(\d+)", r"number of disconnected regions\s*[=:]\s*(\d+)"),
            "disconnected_regions",
            int,
        )
    )
    illegal_faces = _illegal_face_count(diagnostics_text, diagnostics_mesh_ok)
    concave_cells = _concave_cell_count(diagnostics_text)
    case_dir = log_path.parent.resolve()
    try:
        profile = MeshProfile(case_dir.name)
    except ValueError as error:
        raise ValueError(f"cannot infer mesh profile from checkMesh log path: {log_path}") from error
    surface_face_count = _parse_boundary_face_count(case_dir)
    faces_with_layers, layer_coverage, minimum_layers, maximum_layers = _parse_layer_metrics(
        case_dir, case_dir / "log.snappyHexMesh", surface_face_count
    )
    return MeshResult(
        profile=profile,
        case_dir=case_dir,
        mesh_sha256=_mesh_digest(case_dir),
        cell_count=cell_count,
        surface_face_count=surface_face_count,
        layer_coverage_fraction=layer_coverage,
        max_non_orthogonality=max_non_orthogonality,
        max_skewness=max_skewness,
        diagnostic_min_wellposedness_determinant=diagnostic_min_wellposedness_determinant,
        min_volume=min_volume,
        check_mesh_passed=exit_code == 0 and mesh_ok,
        illegal_faces=illegal_faces,
        disconnected_regions=disconnected_regions,
        faces_with_layers=faces_with_layers,
        minimum_layer_count=minimum_layers,
        maximum_layer_count=maximum_layers,
        mesh_ok=mesh_ok,
        all_geometry_passed=diagnostics_exit_code == 0 and diagnostics_mesh_ok,
        concave_cells=concave_cells,
    )


def _process_tree_rss_bytes(root_pid: int) -> int:
    parents: dict[int, int] = {}
    rss: dict[int, int] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat_path.read_text(encoding="utf-8").split()
            pid = int(fields[0])
            parents[pid] = int(fields[3])
            status = stat_path.with_name("status").read_text(encoding="utf-8")
            match = re.search(r"^VmRSS:\s+(\d+)\s+kB", status, flags=re.MULTILINE)
            rss[pid] = 0 if match is None else int(match.group(1)) * 1024
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(rss.get(pid, 0) for pid in descendants)


def _run_command(argv: Sequence[str | Path], case_dir: Path, log_path: Path) -> dict[str, int | float]:
    command = tuple(str(value) for value in argv)
    started_at = datetime.now(UTC).isoformat()
    start_free = shutil.disk_usage(case_dir).free
    minimum_free = start_free
    peak_rss = 0
    output: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"started_at={started_at}\ncwd={case_dir}\nargv={command!r}\n\n")
        log.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=case_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            log.write(f"launch_error={error}\nended_at={datetime.now(UTC).isoformat()}\nexit_status=127\n")
            log.flush()
            os.fsync(log.fileno())
            raise CommandFailure(command, 127, log_path) from error
        assert process.stdout is not None
        for line in process.stdout:
            output.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            peak_rss = max(peak_rss, _process_tree_rss_bytes(process.pid))
            minimum_free = min(minimum_free, shutil.disk_usage(case_dir).free)
        returncode = process.wait()
        peak_rss = max(peak_rss, _process_tree_rss_bytes(process.pid))
        minimum_free = min(minimum_free, shutil.disk_usage(case_dir).free)
        log.write(f"\nended_at={datetime.now(UTC).isoformat()}\nexit_status={returncode}\n")
        log.flush()
        os.fsync(log.fileno())
    if returncode != 0:
        raise CommandFailure(command, returncode, log_path)
    return {
        "exit_code": returncode,
        "peak_rss_bytes": peak_rss,
        "start_free_bytes": start_free,
        "minimum_free_bytes": minimum_free,
        "peak_disk_consumption_bytes": max(0, start_free - minimum_free),
    }


def audit_surface(case_dir: Path) -> Path:
    validated_case = Path(case_dir).resolve(strict=True)
    surface = (validated_case / "constant" / "triSurface" / "hull.stl").resolve(strict=True)
    if surface.parent.parent.parent != validated_case:
        raise ValueError(f"mesh surface escapes validated case directory: {surface}")
    log = validated_case / "log.surfaceCheck"
    _run_command(
        [require_executable("surfaceCheck"), "-checkSelfIntersection", "-writeSets", "vtk", surface],
        validated_case,
        log,
    )
    text = log.read_text(encoding="utf-8", errors="replace")
    missing = [proof for proof in _SURFACE_AUDIT_LINES if proof not in text]
    if missing:
        raise ValueError(f"surface audit failed; missing proofs {missing}; log: {log}")
    return log


def _guarded_cleanup(case_dir: Path) -> list[str]:
    validated_case = case_dir.resolve(strict=True)
    if validated_case.parent.name != "cases":
        raise ValueError(f"mesh case is not below an active run cases directory: {validated_case}")
    run_root = validated_case.parent.parent.resolve(strict=True)
    if run_root.parent.name != "runs":
        raise ValueError(f"mesh case is outside the active run root: {validated_case}")
    removed: list[str] = []
    for candidate in sorted(validated_case.glob("processor*")):
        if candidate.is_symlink():
            raise ValueError(f"refusing processor cleanup through symlink: {candidate}")
        resolved = candidate.resolve(strict=True)
        if resolved.parent != validated_case:
            raise ValueError(f"refusing processor cleanup outside validated case: {resolved}")
        try:
            resolved.relative_to(run_root)
        except ValueError as error:
            raise ValueError(f"refusing processor cleanup outside active run root: {resolved}") from error
        if not resolved.is_dir():
            raise ValueError(f"refusing processor cleanup of non-directory: {resolved}")
        shutil.rmtree(resolved)
        removed.append(str(resolved))
    return removed


def _surface_deviation(mesh_surface: trimesh.Trimesh, reference_surface: trimesh.Trimesh) -> float:
    mesh_points = np.asarray(mesh_surface.vertices, dtype=float)
    reference_points = np.asarray(reference_surface.vertices, dtype=float)
    if len(mesh_points) == 0 or len(reference_points) == 0:
        raise ValueError("surface deviation requires non-empty surfaces")
    # STL vertices lie on the validated source triangles. A nearest-source-vertex
    # query is deterministic and conservative relative to point-to-triangle distance.
    from scipy.spatial import cKDTree

    tree = cKDTree(reference_points)
    distances, _ = tree.query(mesh_points, k=1, workers=1)
    maximum = float(np.max(distances))
    if not math.isfinite(maximum):
        raise ValueError("surface deviation is non-finite")
    return maximum


def _read_ascii_vtu_points(path: Path) -> np.ndarray:
    for _, element in ET.iterparse(path, events=("end",)):
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "DataArray" and element.attrib.get("Name") == "Points":
            if element.attrib.get("format") != "ascii":
                raise ValueError(f"VTK point array is not ASCII: {path}")
            if int(element.attrib.get("NumberOfComponents", "0")) != 3:
                raise ValueError(f"VTK point array is not three-dimensional: {path}")
            values = np.fromstring(element.text or "", sep=" ", dtype=float)
            if values.size == 0 or values.size % 3:
                raise ValueError(f"VTK point array has an invalid size: {path}")
            return values.reshape((-1, 3))
        element.clear()
    raise ValueError(f"VTK file lacks a named Points array: {path}")


def _render_figures(case_dir: Path, profile: MeshProfile) -> tuple[Path, Path, float, dict[str, str]]:
    figures = case_dir.parent.parent / "meshes" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    requested_surface = figures / f"{profile.value}-hull-mesh.stl"
    surface_log = case_dir / "log.surfaceMeshExtract"
    _run_command(
        [require_executable("surfaceMeshExtract"), "-case", case_dir, "-constant", "-patches", "(hull)", requested_surface],
        case_dir,
        surface_log,
    )
    exported_surface = requested_surface
    if not exported_surface.is_file():
        exported_surface = requested_surface.with_name(f"{requested_surface.stem}_0{requested_surface.suffix}")
    if not exported_surface.is_file():
        raise ValueError(f"surfaceMeshExtract did not create the requested hull surface: {requested_surface}")
    vtk_name = f"VTK-mesh-{profile.value}"
    vtk_log = case_dir / "log.foamToVTK"
    _run_command(
        [require_executable("foamToVTK"), "-case", case_dir, "-constant", "-no-fields", "-ascii", "-name", vtk_name, "-overwrite"],
        case_dir,
        vtk_log,
    )
    vtk_root = case_dir / vtk_name
    internal_candidates = sorted(vtk_root.rglob("internal.vtu")) or sorted(vtk_root.rglob("*.vtu"))
    if not internal_candidates:
        raise ValueError(f"foamToVTK did not create an internal volume mesh under {vtk_root}")
    points = _read_ascii_vtu_points(internal_candidates[-1])
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        raise ValueError("VTK volume mesh lacks three-dimensional points")
    y_span = float(np.ptp(points[:, 1]))
    tolerance = max(y_span / 500.0, 1.0e-9)
    selected = points[np.abs(points[:, 1]) <= tolerance]
    if len(selected) < 20:
        order = np.argsort(np.abs(points[:, 1]))[: min(5000, len(points))]
        selected = points[order]
    slice_png = figures / f"{profile.value}-longitudinal-slice.png"
    fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    axis.scatter(selected[:, 0], selected[:, 2], s=0.15, color="#124e78", rasterized=True)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("z [m]")
    axis.set_title(f"{profile.value.title()} reconstructed mesh, longitudinal centre slice")
    fig.savefig(slice_png, dpi=180)
    plt.close(fig)

    mesh_surface = trimesh.load(exported_surface, process=False, maintain_order=True)
    reference_surface = trimesh.load(case_dir / "constant" / "triSurface" / "hull.stl", process=False, maintain_order=True)
    if not isinstance(mesh_surface, trimesh.Trimesh) or not isinstance(reference_surface, trimesh.Trimesh):
        raise ValueError("surface export did not produce one triangulated hull")
    deviation = _surface_deviation(mesh_surface, reference_surface)
    vertices = np.asarray(mesh_surface.vertices, dtype=float)
    surface_png = figures / f"{profile.value}-surface-cells.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(vertices[:, 0], vertices[:, 1], s=0.12, color="#b5442b", rasterized=True)
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("plan view")
    axes[1].scatter(vertices[:, 0], vertices[:, 2], s=0.12, color="#b5442b", rasterized=True)
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("z [m]")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title("profile view")
    fig.suptitle(f"{profile.value.title()} hull surface-cell vertices")
    fig.savefig(surface_png, dpi=180)
    plt.close(fig)
    return slice_png, surface_png, deviation, {
        "surfaceMeshExtract": str(surface_log),
        "foamToVTK": str(vtk_log),
        "vtk_internal": str(internal_candidates[-1]),
        "exported_surface": str(exported_surface),
    }


def generate_mesh(
    case_dir: Path,
    profile: MeshProfile,
    runtime: OpenFoamRuntime,
    config: PipelineConfig,
) -> MeshResult:
    validated_case = Path(case_dir).resolve(strict=True)
    profile = MeshProfile(profile)
    if validated_case.name != profile.value:
        raise ValueError(f"mesh profile {profile.value} does not match validated case {validated_case}")
    ranks = config.raw["mpi_ranks"]
    if isinstance(ranks, bool) or not isinstance(ranks, int) or not 1 <= ranks <= 8:
        raise ValueError("mesh MPI ranks must be an integer from 1 through 8")
    audit_log = audit_surface(validated_case)
    resources: dict[str, object] = {"surfaceCheck": {"log": str(audit_log)}}
    removed: list[str] = []
    reconstructed = False
    parsed: MeshResult | None = None
    parsing_error: ValueError | None = None
    try:
        for name, raw in _COMMANDS:
            values = tuple(str(ranks) if value == "{ranks}" else value for value in raw)
            if name == "snappyHexMesh":
                argv: list[str | Path] = [runtime.mpirun, *values[1:4], values[4], values[5], values[6], require_executable(values[7]), *values[8:], "-case", validated_case]
            else:
                argv = [require_executable(values[0]), "-case", validated_case, *values[1:]]
            log = validated_case / f"log.{name}"
            resources[name] = {"log": str(log), **_run_command(argv, validated_case, log)}
            reconstructed = reconstructed or name == "reconstructParMesh"
        try:
            parsed = parse_check_mesh(
                validated_case / "log.checkMesh",
                validated_case / "log.checkMeshAllGeometry",
            )
        except ValueError as error:
            parsing_error = error
    finally:
        if reconstructed:
            removed = _guarded_cleanup(validated_case)
    if parsing_error is not None:
        atomic_write_json(
            validated_case / "mesh-result.json",
            {
                "schema_version": 2,
                "result": None,
                "accepted": False,
                "parse_error": str(parsing_error),
                "resources": resources,
                "processor_directories_removed": removed,
            },
        )
        raise parsing_error
    if parsed is None:
        raise RuntimeError("mesh parsing ended without a result or a recorded error")
    try:
        slice_png, surface_png, deviation, figure_logs = _render_figures(validated_case, profile)
    except (OSError, RuntimeError, ValueError) as error:
        atomic_write_json(
            validated_case / "mesh-result.json",
            {
                "schema_version": 2,
                "result": None,
                "observed_result": _as_json(asdict(parsed)),
                "accepted": False,
                "rejection_code": "FIGURE_GENERATION_FAILED",
                "figure_error": str(error),
                "resources": resources,
                "processor_directories_removed": removed,
            },
        )
        raise
    mesh_config = config.raw["mesh"]
    accepted = (
        parsed.check_mesh_passed
        and parsed.mesh_ok
        and parsed.min_volume > 0.0
        and parsed.max_non_orthogonality <= float(mesh_config["max_non_orthogonality"])
        and parsed.max_skewness <= float(mesh_config["max_internal_skewness"])
        and parsed.illegal_faces == 0
        and parsed.disconnected_regions == 1
        and parsed.layer_coverage_fraction >= float(mesh_config["minimum_layer_coverage_fraction"])
    )
    result = replace(parsed, check_mesh_passed=accepted, final_surface_deviation_m=deviation)
    payload = {
        "schema_version": 2,
        "result": _as_json(asdict(result)),
        "accepted": accepted,
        "quality_thresholds": {
            "minimum_determinant": mesh_config["minimum_determinant"],
            "max_non_orthogonality": mesh_config["max_non_orthogonality"],
            "max_internal_skewness": mesh_config["max_internal_skewness"],
            "minimum_layer_coverage_fraction": mesh_config["minimum_layer_coverage_fraction"],
        },
        "diagnostics": {
            "all_geometry_passed": result.all_geometry_passed,
            "minimum_wellposedness_determinant": result.diagnostic_min_wellposedness_determinant,
            "concave_cells": result.concave_cells,
            "policy": (
                "OpenFOAM documents -allGeometry as including non-finite-volume-specific checks. "
                "Its wellposedness determinant and concave-cell count are diagnostic only; "
                "CFD acceptance is decided by checkMesh -meshQuality and the explicit finite-volume gates."
            ),
        },
        "resources": resources,
        "processor_directories_removed": removed,
        "figures": {"longitudinal_slice": str(slice_png), "surface_cells": str(surface_png)},
        "figure_sources": figure_logs,
    }
    atomic_write_json(validated_case / "mesh-result.json", payload)
    if not accepted:
        raise ValueError(f"mesh quality gates rejected {profile.value}; evidence: {validated_case / 'mesh-result.json'}")
    return result


def mesh_result_from_json(path: Path) -> MeshResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("result")
    if not isinstance(value, dict):
        raise ValueError(f"invalid mesh result: {path}")
    value = dict(value)
    value["profile"] = MeshProfile(value["profile"])
    value["case_dir"] = Path(value["case_dir"]).resolve(strict=True)
    result = MeshResult(**value)
    if not result.check_mesh_passed:
        raise ValueError(f"mesh result is not accepted: {path}")
    return result
