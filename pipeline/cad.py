from __future__ import annotations

import contextlib
import hashlib
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import gmsh
import matplotlib
import numpy as np
import trimesh
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import Polygon
from shapely.ops import unary_union

from pipeline.models import GeometryResult, OpenFoamRuntime, PipelineConfig, RunPaths
from pipeline.orientation import OrientationResult, detect_orientation, normalize_geometry
from pipeline.runtime import run_checked
from pipeline.state import atomic_write_json, sha256_file


UNIT_TO_METRES = {
    "METRE": 1.0,
    "MILLI_METRE": 1.0e-3,
    "CENTI_METRE": 1.0e-2,
    "INCH": 0.0254,
}
_ENTITY = re.compile(r"#(?P<id>\d+)\s*=\s*(?P<body>.*?);", re.DOTALL)
_SI_UNIT = re.compile(
    r"SI_UNIT\s*\(\s*(?:\.([A-Z_]+)\.|[$*])\s*,\s*\.([A-Z_]+)\.\s*\)",
    re.DOTALL,
)
_CONVERSION = re.compile(r"CONVERSION_BASED_UNIT\s*\(\s*'([^']+)'", re.IGNORECASE | re.DOTALL)
_POINT = re.compile(r"CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(([^)]*)\)\s*\)", re.DOTALL)


def _length_unit_matches(source: Path) -> list[tuple[str, float, str]]:
    text = source.read_text(encoding="utf-8", errors="strict")
    matches: list[tuple[str, float, str]] = []
    for entity in _ENTITY.finditer(text):
        body = entity.group("body")
        if "LENGTH_UNIT" not in body:
            continue
        si = _SI_UNIT.search(body)
        conversion = _CONVERSION.search(body)
        if si is not None:
            prefix, base = si.groups()
            name = f"{prefix}_{base}" if prefix is not None else base
        elif conversion is not None:
            name = conversion.group(1).strip().upper().replace(" ", "_")
        else:
            raise ValueError(f"LENGTH_UNIT entity #{entity.group('id')} has no supported SI or conversion unit")
        if name not in UNIT_TO_METRES:
            raise ValueError(f"unsupported STEP length unit {name!r} in entity #{entity.group('id')}")
        matches.append((name, UNIT_TO_METRES[name], entity.group(0)))
    if not matches:
        raise ValueError("STEP file does not declare a LENGTH_UNIT complex entity")
    distinct = {(name, scale) for name, scale, _ in matches}
    if len(distinct) != 1:
        raise ValueError("STEP file declares contradictory length units")
    return matches


def read_step_unit(path: Path) -> tuple[str, float]:
    matches = _length_unit_matches(path)
    return matches[0][0], matches[0][1]


def _step_point_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    text = path.read_text(encoding="utf-8", errors="strict")
    points: list[tuple[float, float, float]] = []
    for match in _POINT.finditer(text):
        values = [value.strip().replace("D", "E") for value in match.group(1).split(",")]
        if len(values) != 3:
            continue
        try:
            points.append(tuple(float(value) for value in values))
        except ValueError:
            continue
    if not points:
        raise ValueError("STEP file has no parseable CARTESIAN_POINT coordinates")
    array = np.asarray(points, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("STEP Cartesian coordinates are not finite")
    return array.min(axis=0), array.max(axis=0)


@contextlib.contextmanager
def gmsh_session() -> Iterator[None]:
    gmsh.initialize()
    try:
        yield
    finally:
        gmsh.finalize()


def tessellate_step(source: Path, destination: Path, chord_fraction: float, angle_deg: float) -> None:
    if not 0.0 < chord_fraction < 1.0:
        raise ValueError("STEP chord fraction must be between zero and one")
    if not 0.0 < angle_deg <= 180.0:
        raise ValueError("STEP angle must be in (0, 180]")
    destination.parent.mkdir(parents=True, exist_ok=True)
    def configure_import() -> None:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
        gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
        gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
        gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
        gmsh.option.setNumber("Geometry.OCCMakeSolids", 1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", max(12, math.ceil(360.0 / angle_deg)))
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    with gmsh_session():
        configure_import()
        gmsh.model.add("healed_hull")
        gmsh.model.occ.importShapes(str(source))
        gmsh.model.occ.synchronize()
        gmsh.model.occ.healShapes()
        gmsh.model.occ.synchronize()

        # OpenCASCADE can sew a valid single shell yet fail its automatic
        # MakeSolid step. Promote that healed shell explicitly instead of
        # falling back to a second unhealed import.
        healed_volumes = gmsh.model.getEntities(3)
        if not healed_volumes:
            healed_surfaces = [tag for dimension, tag in gmsh.model.getEntities(2) if dimension == 2]
            if not healed_surfaces:
                raise ValueError("STEP healing yielded neither a volume nor a surface shell")
            surface_loop = gmsh.model.occ.addSurfaceLoop(healed_surfaces, sewing=True)
            promoted_volume = gmsh.model.occ.addVolume([surface_loop])
            gmsh.model.occ.synchronize()
            volume_surfaces = set(
                gmsh.model.getBoundary([(3, promoted_volume)], combined=False, oriented=False, recursive=False)
            )
            orphan_surfaces = set(gmsh.model.getEntities(2)) - volume_surfaces
            if orphan_surfaces:
                gmsh.model.occ.remove(sorted(orphan_surfaces), recursive=False)
                gmsh.model.occ.synchronize()
            healed_volumes = gmsh.model.getEntities(3)
        if len(healed_volumes) != 1:
            raise ValueError(f"STEP healing must yield exactly one volume; found {len(healed_volumes)}")
        volume_tag = healed_volumes[0][1]
        bounds = gmsh.model.getBoundingBox(3, volume_tag)
        extents = np.asarray((bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2]), dtype=float)
        max_extent = float(extents.max())
        if not math.isfinite(max_extent) or max_extent <= 0.0:
            raise ValueError("healed STEP volume has invalid metre-space bounds")
        _, scale_to_m = read_step_unit(source)
        parsed_min, parsed_max = _step_point_bounds(source)
        parsed_extents = (parsed_max - parsed_min) * scale_to_m
        relative_error = np.max(np.abs(extents - parsed_extents) / np.maximum(parsed_extents, 1.0e-12))
        if float(relative_error) > 0.001:
            raise ValueError(
                f"Gmsh metre-space extent disagrees with parsed STEP extent by {relative_error:.3%} (limit 0.1%)"
            )
        chord_m = max_extent * chord_fraction
        gmsh.option.setNumber("Mesh.MeshSizeMin", chord_m * 0.75)
        gmsh.option.setNumber("Mesh.MeshSizeMax", chord_m * 2.0)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.removeDuplicateNodes()
        removed_degenerate = False
        boundary_surfaces = gmsh.model.getBoundary(
            [(3, volume_tag)], combined=False, oriented=False, recursive=False
        )
        for _, surface_tag in boundary_surfaces:
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(2, surface_tag)
            for element_type, tags, nodes in zip(element_types, element_tags, node_tags, strict=True):
                _, _, _, nodes_per_element, _, primary_nodes = gmsh.model.mesh.getElementProperties(element_type)
                connectivity = np.asarray(nodes, dtype=np.int64).reshape((-1, nodes_per_element))
                degenerate = np.asarray(
                    [len(set(row[:primary_nodes])) < primary_nodes for row in connectivity], dtype=bool
                )
                if np.any(degenerate):
                    gmsh.model.mesh.removeElements(2, surface_tag, np.asarray(tags)[degenerate].tolist())
                    removed_degenerate = True
        if removed_degenerate:
            gmsh.model.mesh.reclassifyNodes()
        gmsh.model.mesh.setOutwardOrientation(volume_tag)
        gmsh.write(str(destination))


def load_validated_surface(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False, maintain_order=True)
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if not geometries or not all(isinstance(item, trimesh.Trimesh) for item in geometries):
            raise ValueError("STL scene does not contain only mesh geometries")
        mesh = trimesh.util.concatenate(geometries)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"unsupported triangulated surface type: {type(loaded).__name__}")
    with path.open("rb") as stream:
        prefix = stream.read(512)
    if path.suffix.lower() == ".stl" and prefix.lstrip().startswith(b"solid") and b"facet" in prefix:
        # ASCII STL repeats each facet's coordinates. Canonicalize only
        # numerically identical coordinate triples: no rounding, tolerance,
        # nearby-point welding, face removal, or hole repair is permitted.
        vertices = np.asarray(mesh.vertices, dtype=float)
        canonical_vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
        if len(canonical_vertices) != len(vertices):
            mesh = trimesh.Trimesh(
                vertices=canonical_vertices,
                faces=inverse[np.asarray(mesh.faces, dtype=np.int64)],
                process=False,
            )
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    graph = coo_matrix(
        (np.ones(adjacency.size, dtype=np.uint8), (adjacency.ravel(), adjacency[:, ::-1].ravel())),
        shape=(len(mesh.faces), len(mesh.faces)),
    )
    component_count, _ = connected_components(graph, directed=False, return_labels=True)
    if component_count != 1:
        raise ValueError("triangulated surface is not a single connected solid")
    if not mesh.is_watertight:
        raise ValueError("triangulated surface is not watertight")
    if not mesh.is_winding_consistent:
        raise ValueError("triangulated surface winding is inconsistent")
    if not mesh.is_volume:
        raise ValueError("triangulated surface does not enclose a consistently oriented volume")
    if not math.isfinite(float(mesh.area)) or float(mesh.area) <= 0.0:
        raise ValueError("triangulated surface has non-positive area")
    if not math.isfinite(float(mesh.volume)) or abs(float(mesh.volume)) <= 0.0:
        raise ValueError("triangulated surface has non-positive volume")
    extent = float(np.max(mesh.extents))
    if not 0.05 <= extent <= 20.0:
        raise ValueError(f"triangulated surface maximum dimension {extent:.6g} m is outside [0.05, 20] m")
    return mesh


def _frontal_area(mesh: trimesh.Trimesh) -> float:
    polygons: list[Polygon] = []
    for triangle in np.asarray(mesh.triangles, dtype=float):
        polygon = Polygon(triangle[:, 1:3])
        if polygon.is_empty or polygon.area <= 1.0e-15:
            continue
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 1.0e-15:
            raise ValueError("triangle projection produced an invalid frontal polygon")
        polygons.append(polygon)
    if not polygons:
        raise ValueError("triangulated surface has no non-degenerate frontal projections")
    union = unary_union(polygons)
    if union.is_empty or not union.is_valid or union.area <= 0.0:
        raise ValueError("frontal silhouette union is empty or invalid")
    return float(union.area)


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _preview(mesh: trimesh.Trimesh, orientation: OrientationResult, output: Path) -> None:
    matplotlib.use("Agg", force=True)
    figure = plt.figure(figsize=(16, 9), dpi=100)
    axis = figure.add_subplot(111, projection="3d")
    vertices = np.asarray(mesh.vertices, dtype=float)
    axis.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=mesh.faces, color="#8fbcd4", alpha=0.72, linewidth=0.05)
    center = np.asarray(mesh.center_mass, dtype=float)
    bow = np.asarray(orientation.bow_vector, dtype=float)
    flow = np.asarray(orientation.relative_flow_vector, dtype=float)
    length = float(np.max(mesh.extents))
    bow_point = center + bow * length * 0.5
    stern_point = center - bow * length * 0.5
    axis.scatter(*bow_point, color="#d62728", s=55, label="bow")
    axis.scatter(*stern_point, color="#2ca02c", s=55, label="stern")
    axis.quiver(*center, *(flow * length * 0.32), color="#1f77b4", linewidth=2.5, label="relative flow")
    for index, color in enumerate(("#d62728", "#2ca02c", "#1f77b4")):
        direction = np.zeros(3)
        direction[index] = length * 0.2
        axis.quiver(*center, *direction, color=color, alpha=0.65)
    axis.set_xlabel("original X (m)")
    axis.set_ylabel("original Y (m)")
    axis.set_zlabel("original Z (m)")
    axis.set_title(f"Healed hull: bow {orientation.original_axis_label}, confidence {orientation.confidence:.3f}")
    axis.legend(loc="upper right")
    axis.set_box_aspect(tuple(float(value) for value in mesh.extents))
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def prepare_geometry(
    source: Path,
    paths: RunPaths,
    config: PipelineConfig,
    runtime: OpenFoamRuntime,
    bow_override: str | None,
) -> GeometryResult:
    source = source.resolve()
    declared_unit, scale_to_m = read_step_unit(source)
    unit_entity = _length_unit_matches(source)[0][2]
    source_sha256 = sha256_file(source)
    original_stl = paths.geometry / "original.stl"
    normalized_stl = paths.geometry / "hull.stl"
    orientation_path = paths.geometry / "orientation.json"
    preview_path = paths.geometry / "orientation.png"
    tessellate_step(
        source,
        original_stl,
        float(config.raw["geometry"]["step_chord_fraction"]),
        float(config.raw["geometry"]["step_angle_deg"]),
    )
    original = load_validated_surface(original_stl)
    parsed_min, parsed_max = _step_point_bounds(source)
    parsed_extents = (parsed_max - parsed_min) * scale_to_m
    tessellated_extents = np.asarray(original.extents, dtype=float)
    tessellated_relative_error = np.max(
        np.abs(tessellated_extents - parsed_extents) / np.maximum(parsed_extents, 1.0e-12)
    )
    orientation = detect_orientation(original, config, bow_override)
    normalized = normalize_geometry(original, orientation)
    preclean_stl = paths.geometry / "hull.pre-clean.stl"
    normalized.export(preclean_stl, file_type="stl")
    surface_clean_log = paths.geometry / "surfaceClean.log"
    cleanup_length_m = float(np.max(normalized.extents)) * 3.0e-7
    run_checked(
        ["surfaceClean", str(preclean_stl), f"{cleanup_length_m:.12g}", "1e-10", str(normalized_stl)],
        paths.geometry,
        surface_clean_log,
    )
    normalized = load_validated_surface(normalized_stl)
    frontal_area = _frontal_area(normalized)
    _preview(original, orientation, preview_path)
    flow = np.asarray(orientation.relative_flow_vector, dtype=float)
    rotation = trimesh.geometry.align_vectors(flow, np.array((1.0, 0.0, 0.0)))
    translation = np.eye(4)
    translation[:3, 3] = -np.asarray(original.center_mass, dtype=float)
    transform = rotation @ translation
    inverse = np.linalg.inv(transform)
    orientation_payload = {
        "source": str(source),
        "source_sha256": source_sha256,
        "declared_unit": declared_unit,
        "scale_to_m": scale_to_m,
        "matched_step_unit_entity": unit_entity,
        "parsed_step_bounds_native": {"min": [float(value) for value in parsed_min], "max": [float(value) for value in parsed_max]},
        "tessellated_bounds_m": {"min": [float(value) for value in original.bounds[0]], "max": [float(value) for value in original.bounds[1]]},
        "tessellated_vs_parsed_extent_relative_error": float(tessellated_relative_error),
        "orientation": orientation.metadata(),
        "transform": _matrix_tuple(transform),
        "inverse_transform": _matrix_tuple(inverse),
        "metrics": {
            "wetted_area_m2": float(normalized.area),
            "volume_m3": abs(float(normalized.volume)),
            "centroid_m": [float(value) for value in normalized.center_mass],
            "frontal_area_m2": frontal_area,
            "frontal_area_algorithm": "shapely_triangle_union",
            "normalized_extents_m": [float(value) for value in normalized.extents],
        },
    }
    atomic_write_json(orientation_path, orientation_payload)
    surface_log = paths.geometry / "surfaceCheck.log"
    run_checked(
        ["surfaceCheck", "-checkSelfIntersection", "-writeSets", "vtk", str(normalized_stl)],
        paths.geometry,
        surface_log,
    )
    audit = surface_log.read_text(encoding="utf-8", errors="replace")
    required = (
        "Surface has no illegal triangles.",
        "Surface is closed. All edges connected to two faces.",
        "Number of unconnected parts : 1",
        "Number of zones (connected area with consistent normal) : 1",
        "Surface is not self-intersecting",
    )
    missing = [line for line in required if line not in audit]
    if missing:
        raise ValueError(f"surfaceCheck did not prove a closed singly-connected non-intersecting surface: missing {missing}")
    return GeometryResult(
        source=source,
        source_sha256=source_sha256,
        declared_unit=declared_unit,
        scale_to_m=scale_to_m,
        normalized_stl=normalized_stl,
        original_bounds_m=tuple(tuple(float(value) for value in row) for row in original.bounds),
        normalized_extents_m=tuple(float(value) for value in normalized.extents),
        length_m=float(normalized.extents[0]),
        frontal_area_m2=frontal_area,
        wetted_area_m2=float(normalized.area),
        volume_m3=abs(float(normalized.volume)),
        centroid_m=tuple(float(value) for value in normalized.center_mass),
        bow_original_axis=orientation.original_axis_label,
        bow_confidence=orientation.confidence,
        transform=_matrix_tuple(transform),
        inverse_transform=_matrix_tuple(inverse),
        original_stl=original_stl,
        orientation_metadata=orientation_path,
        preview_png=preview_path,
        surface_check_log=surface_log,
        step_unit_entity=unit_entity,
    )
