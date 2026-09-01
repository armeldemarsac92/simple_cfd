from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import trimesh

from pipeline.models import PipelineConfig


@dataclass(frozen=True)
class EndSignature:
    cap_area_fraction: float
    early_area_growth: float
    taper_linearity: float
    curvature_score: float


@dataclass(frozen=True)
class OrientationResult:
    original_axis_label: str
    bow_vector: tuple[float, float, float]
    relative_flow_vector: tuple[float, float, float]
    confidence: float
    negative_end: EndSignature
    positive_end: EndSignature
    negative_components: tuple[tuple[str, float], ...]
    positive_components: tuple[tuple[str, float], ...]
    negative_score: float
    positive_score: float
    overridden: bool

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("cannot normalize a zero-length orientation vector")
    return vector / length


def _axis_label(vector: np.ndarray) -> str:
    labels = ("x", "y", "z")
    index = int(np.argmax(np.abs(vector)))
    return ("+" if vector[index] >= 0.0 else "-") + labels[index]


def _canonical_axis(vector: np.ndarray) -> np.ndarray:
    """Make PCA sign reproducible by pointing toward the positive source axis."""
    index = int(np.argmax(np.abs(vector)))
    return vector if vector[index] >= 0.0 else -vector


def _weighted_covariance(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(mesh.triangles_center, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    if centers.shape[0] == 0 or not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError("surface has invalid triangle centers or areas")
    total_area = float(areas.sum())
    mean = np.average(centers, axis=0, weights=areas)
    centered = centers - mean
    covariance = (centered * areas[:, None]).T @ centered / total_area
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def _end_signature(
    mesh: trimesh.Trimesh,
    axis: np.ndarray,
    vertex_projections: np.ndarray,
    face_projections: np.ndarray,
    end: str,
    bins: int,
    terminal_fraction: float,
) -> EndSignature:
    minimum, maximum = float(vertex_projections.min()), float(vertex_projections.max())
    length = maximum - minimum
    if length <= 0.0:
        raise ValueError("principal axis has zero extent")
    face_areas = np.asarray(mesh.area_faces, dtype=float)
    normals = np.asarray(mesh.face_normals, dtype=float)
    if end == "negative":
        distance = face_projections - minimum
        inward_axis = axis
    else:
        distance = maximum - face_projections
        inward_axis = -axis

    terminal = distance <= terminal_fraction * length
    if not np.any(terminal):
        raise ValueError(f"no faces found in the {end} terminal region")
    cap = (distance <= 0.01 * length) & (np.abs(normals @ axis) >= 0.95)
    terminal_area = float(face_areas[terminal].sum())
    cap_fraction = float(face_areas[cap].sum() / terminal_area) if terminal_area > 0.0 else 0.0

    # Face area per axial bin is a stable surface-profile proxy that remains
    # defined for arbitrary closed triangulations (including appendage-free hulls).
    bin_index = np.minimum((distance / length * bins).astype(int), bins - 1)
    profile = np.bincount(bin_index, weights=face_areas, minlength=bins).astype(float)
    terminal_bins = max(2, int(np.ceil(bins * terminal_fraction)))
    local = profile[:terminal_bins]
    split = max(1, terminal_bins // 2)
    early = float(np.mean(local[:split]))
    later = float(np.mean(local[split:]))
    early_growth = float(np.clip((later - early) / max(later, 1.0e-15), 0.0, 1.0))

    x = (np.arange(terminal_bins, dtype=float) + 0.5) / terminal_bins
    active = local > max(float(local.max()) * 1.0e-6, 1.0e-15)
    if int(active.sum()) < 3:
        linearity = 1.0
    else:
        coefficients = np.polyfit(x[active], local[active], 1)
        predicted = np.polyval(coefficients, x[active])
        residual = float(np.square(local[active] - predicted).sum())
        total = float(np.square(local[active] - local[active].mean()).sum())
        linearity = float(np.clip(1.0 - residual / total, 0.0, 1.0)) if total > 1.0e-15 else 1.0

    local_normals = normals[terminal]
    local_weights = face_areas[terminal]
    # A planar cap has a concentrated normal distribution; a rounded nose has
    # a broad distribution even when its net direction remains axial.
    normal_resultant = np.average(local_normals, axis=0, weights=local_weights)
    curvature = float(np.clip(1.0 - np.linalg.norm(normal_resultant), 0.0, 1.0))
    axial_variation = np.average(
        np.square((local_normals @ inward_axis) - np.average(local_normals @ inward_axis, weights=local_weights)),
        weights=local_weights,
    )
    curvature = float(np.clip(0.5 * curvature + 0.5 * np.sqrt(max(float(axial_variation), 0.0)), 0.0, 1.0))
    return EndSignature(cap_fraction, early_growth, linearity, curvature)


def _contrast(
    values: tuple[float, float], prefer_larger: bool, noise_floor: float
) -> tuple[float, float]:
    if not math.isfinite(noise_floor) or noise_floor <= 0.0:
        raise ValueError("orientation contrast noise floor must be positive and finite")
    signed = float(np.clip((values[0] - values[1]) / noise_floor, -1.0, 1.0))
    if not prefer_larger:
        signed = -signed
    return (0.5 + 0.5 * signed, 0.5 - 0.5 * signed)


def _parse_override(override: str | None) -> np.ndarray | None:
    if override is None:
        return None
    labels = {"+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0), "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0), "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0)}
    try:
        return np.asarray(labels[override], dtype=float)
    except KeyError as error:
        raise ValueError(f"invalid bow override: {override}") from error


def detect_orientation(mesh: trimesh.Trimesh, config: PipelineConfig, override: str | None) -> OrientationResult:
    eigenvalues, eigenvectors = _weighted_covariance(mesh)
    axis = _canonical_axis(_unit(eigenvectors[:, 0]))
    vertex_projections = np.asarray(mesh.vertices, dtype=float) @ axis
    face_projections = np.asarray(mesh.triangles_center, dtype=float) @ axis
    extents = np.ptp(np.asarray(mesh.vertices, dtype=float) @ eigenvectors, axis=0)
    transverse = max(float(extents[1]), float(extents[2]))
    dominance = float(extents[0] / transverse) if transverse > 0.0 else float("inf")
    if dominance < float(config.raw["geometry"]["axis_dominance_min"]):
        raise ValueError(
            f"ambiguous longitudinal axis: dominance {dominance:.3f} is below "
            f"{config.raw['geometry']['axis_dominance_min']}"
        )

    terminal_fraction = float(config.raw["geometry"]["terminal_fraction"])
    axial_bins = 40
    contrast_noise_floor = 1.0 / axial_bins
    negative = _end_signature(mesh, axis, vertex_projections, face_projections, "negative", axial_bins, terminal_fraction)
    positive = _end_signature(mesh, axis, vertex_projections, face_projections, "positive", axial_bins, terminal_fraction)
    # For this fixed-hull AUV family the bow is the blunter, more rounded end;
    # the opposite end tapers toward the propulsor. A larger near-end cap
    # fraction and slower early area growth therefore support the bow candidate.
    cap = _contrast(
        (negative.cap_area_fraction, positive.cap_area_fraction),
        prefer_larger=True,
        noise_floor=contrast_noise_floor,
    )
    taper_raw = (
        0.75 * negative.early_area_growth + 0.25 * (1.0 - negative.taper_linearity),
        0.75 * positive.early_area_growth + 0.25 * (1.0 - positive.taper_linearity),
    )
    taper = _contrast(taper_raw, prefer_larger=False, noise_floor=contrast_noise_floor)
    curvature = _contrast(
        (negative.curvature_score, positive.curvature_score),
        prefer_larger=True,
        noise_floor=contrast_noise_floor,
    )
    weights = config.raw["geometry"]
    negative_score = float(weights["cap_weight"] * cap[0] + weights["taper_weight"] * taper[0] + weights["curvature_weight"] * curvature[0])
    positive_score = float(weights["cap_weight"] * cap[1] + weights["taper_weight"] * taper[1] + weights["curvature_weight"] * curvature[1])
    confidence = abs(negative_score - positive_score)

    overridden = override is not None
    requested = _parse_override(override)
    if requested is None:
        if confidence < float(weights["bow_confidence_min"]):
            raise ValueError(
                f"bow confidence {confidence:.3f} is below {weights['bow_confidence_min']}; "
                "supply --bow +x|-x|+y|-y|+z|-z"
            )
        bow = -axis if negative_score > positive_score else axis
    else:
        alignment = float(abs(requested @ axis))
        if alignment < 0.95:
            raise ValueError(
                f"bow override {override} is not aligned with detected longitudinal axis "
                f"({_axis_label(axis)}, alignment={alignment:.3f})"
            )
        bow = requested
    bow = _unit(bow)
    return OrientationResult(
        original_axis_label=_axis_label(bow),
        bow_vector=tuple(float(value) for value in bow),
        relative_flow_vector=tuple(float(value) for value in -bow),
        confidence=float(confidence),
        negative_end=negative,
        positive_end=positive,
        negative_components=(("cap", float(cap[0])), ("taper", float(taper[0])), ("curvature", float(curvature[0]))),
        positive_components=(("cap", float(cap[1])), ("taper", float(taper[1])), ("curvature", float(curvature[1]))),
        negative_score=negative_score,
        positive_score=positive_score,
        overridden=overridden,
    )


def normalize_geometry(mesh: trimesh.Trimesh, orientation: OrientationResult) -> trimesh.Trimesh:
    result = mesh.copy()
    centroid = np.asarray(mesh.center_mass, dtype=float)
    flow = np.asarray(orientation.relative_flow_vector, dtype=float)
    rotation = trimesh.geometry.align_vectors(flow, np.array((1.0, 0.0, 0.0)))
    translation = np.eye(4)
    translation[:3, 3] = -centroid
    transform = rotation @ translation
    result.apply_transform(transform)
    bow = np.asarray(orientation.bow_vector, dtype=float)
    bow_x = float((rotation[:3, :3] @ bow)[0])
    if bow_x >= -0.95:
        raise ValueError("normalization did not place the bow toward negative X")
    if not result.is_watertight or not result.is_winding_consistent or not result.is_volume:
        raise ValueError("normalized surface lost its valid closed-volume properties")
    return result
