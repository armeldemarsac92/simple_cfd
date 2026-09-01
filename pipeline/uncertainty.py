from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq

from pipeline.models import CaseResult, GridStudyResult, MeshProfile, MeshResult, PipelineConfig


@dataclass(frozen=True)
class _ComponentEvaluation:
    classification: str
    fine: float
    medium: float
    coarse: float
    epsilon21: float
    epsilon32: float
    convergence_ratio: float | None
    observed_order: float | None
    extrapolated: float | None
    fine_gci_percent: float | None
    coarse_medium_change_percent: float
    medium_fine_change_percent: float
    reason: str


def _relative_change(first: float, second: float) -> float:
    denominator = abs(second)
    if denominator == 0.0:
        return math.inf
    return abs(first - second) / denominator * 100.0


def _component(fine: float, medium: float, coarse: float, r21: float, r32: float) -> _ComponentEvaluation:
    values = (float(fine), float(medium), float(coarse))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("grid-study drag components must be finite")
    epsilon21 = medium - fine
    epsilon32 = coarse - medium
    scale = max(*(abs(value) for value in values), 1.0)
    tolerance = 1.0e-12 * scale
    coarse_medium_change = _relative_change(coarse, medium)
    medium_fine_change = _relative_change(medium, fine)
    if abs(epsilon21) <= tolerance or abs(epsilon32) <= tolerance:
        return _ComponentEvaluation(
            "degenerate",
            fine,
            medium,
            coarse,
            epsilon21,
            epsilon32,
            None,
            None,
            None,
            None,
            coarse_medium_change,
            medium_fine_change,
            "one or both grid differences are numerically zero",
        )
    ratio = epsilon21 / epsilon32
    if ratio < 0.0:
        classification = "oscillatory_convergence"
        reason = "successive grid differences change sign"
    elif 0.0 < ratio < 1.0:
        classification = "monotonic_convergence"
        reason = "successive grid differences decrease monotonically"
    else:
        classification = "divergence"
        reason = "fine-grid difference does not contract relative to the coarse-grid difference"
    if classification != "monotonic_convergence":
        return _ComponentEvaluation(
            classification,
            fine,
            medium,
            coarse,
            epsilon21,
            epsilon32,
            ratio,
            None,
            None,
            None,
            coarse_medium_change,
            medium_fine_change,
            reason,
        )

    target = abs(epsilon32 / epsilon21)

    def equation(order: float) -> float:
        numerator = r21**order * (r32**order - 1.0)
        denominator = r21**order - 1.0
        return target - numerator / denominator

    try:
        lower = equation(0.1)
        upper = equation(10.0)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower * upper > 0.0:
            raise ValueError("generalized observed-order equation is not bracketed on [0.1, 10]")
        observed_order = float(brentq(equation, 0.1, 10.0, xtol=1.0e-12, rtol=1.0e-12))
        refinement_term = r21**observed_order - 1.0
        if refinement_term <= 0.0 or fine == 0.0:
            raise ValueError("generalized observed-order result has an invalid denominator")
        extrapolated = fine + (fine - medium) / refinement_term
        fine_gci = 1.25 * abs(fine - medium) / (abs(fine) * refinement_term) * 100.0
        if not all(math.isfinite(value) for value in (observed_order, extrapolated, fine_gci)):
            raise ValueError("generalized grid-study result is non-finite")
    except ValueError as error:
        return _ComponentEvaluation(
            classification,
            fine,
            medium,
            coarse,
            epsilon21,
            epsilon32,
            ratio,
            None,
            None,
            None,
            coarse_medium_change,
            medium_fine_change,
            str(error),
        )
    return _ComponentEvaluation(
        classification,
        fine,
        medium,
        coarse,
        epsilon21,
        epsilon32,
        ratio,
        observed_order,
        extrapolated,
        fine_gci,
        coarse_medium_change,
        medium_fine_change,
        reason,
    )


def _component_payload(value: _ComponentEvaluation) -> dict[str, float | str | None]:
    return {
        "classification": value.classification,
        "fine": value.fine,
        "medium": value.medium,
        "coarse": value.coarse,
        "epsilon21": value.epsilon21,
        "epsilon32": value.epsilon32,
        "convergence_ratio": value.convergence_ratio,
        "observed_order": value.observed_order,
        "extrapolated": value.extrapolated,
        "fine_gci_percent": value.fine_gci_percent,
        "coarse_medium_change_percent": value.coarse_medium_change_percent,
        "medium_fine_change_percent": value.medium_fine_change_percent,
        "reason": value.reason,
    }


def evaluate_grid_study(
    fine: CaseResult,
    medium: CaseResult,
    coarse: CaseResult,
    fine_mesh: MeshResult,
    medium_mesh: MeshResult,
    coarse_mesh: MeshResult,
    config: PipelineConfig,
) -> GridStudyResult:
    ordered = (
        (fine, fine_mesh, MeshProfile.FINE),
        (medium, medium_mesh, MeshProfile.MEDIUM),
        (coarse, coarse_mesh, MeshProfile.COARSE),
    )
    for case, mesh, expected in ordered:
        if case.mesh_profile != expected or mesh.profile != expected:
            raise ValueError(f"grid-study profile order mismatch at {expected.value}")
        if case.mesh_sha256 != mesh.mesh_sha256:
            raise ValueError(f"grid-study case/mesh digest mismatch at {expected.value}")
    r21 = (fine_mesh.cell_count / medium_mesh.cell_count) ** (1.0 / 3.0)
    r32 = (medium_mesh.cell_count / coarse_mesh.cell_count) ** (1.0 / 3.0)
    if not math.isfinite(r21) or not math.isfinite(r32) or r21 <= 1.0 or r32 <= 1.0:
        raise ValueError("grid-study effective refinement ratios must be finite and greater than one")

    components = {
        "total": _component(fine.drag_n, medium.drag_n, coarse.drag_n, r21, r32),
        "pressure": _component(
            fine.force_pressure_n[0],
            medium.force_pressure_n[0],
            coarse.force_pressure_n[0],
            r21,
            r32,
        ),
        "viscous": _component(
            fine.force_viscous_n[0],
            medium.force_viscous_n[0],
            coarse.force_viscous_n[0],
            r21,
            r32,
        ),
    }
    total = components["total"]
    maximum_gci = float(config.raw["qualification"]["maximum_fine_gci_percent"])
    rejected_cases = [case.mesh_profile.value for case in (fine, medium, coarse) if not case.accepted]
    if rejected_cases:
        accepted = False
        reason = "rejected reference cases: " + ", ".join(rejected_cases)
    elif total.classification != "monotonic_convergence":
        accepted = False
        reason = f"total drag classification is {total.classification}"
    elif total.observed_order is None or total.fine_gci_percent is None:
        accepted = False
        reason = "total drag generalized observed order or GCI is unavailable"
    elif total.fine_gci_percent > maximum_gci:
        accepted = False
        reason = f"fine-grid GCI {total.fine_gci_percent:.6g}% exceeds {maximum_gci:.6g}%"
    else:
        accepted = True
        reason = (
            f"monotonic total-drag convergence with p={total.observed_order:.6g} and "
            f"fine-grid GCI={total.fine_gci_percent:.6g}%"
        )
    return GridStudyResult(
        classification=total.classification,
        effective_r21=r21,
        effective_r32=r32,
        observed_order=total.observed_order,
        extrapolated_drag_n=total.extrapolated,
        fine_gci_percent=total.fine_gci_percent,
        coarse_medium_change_percent=total.coarse_medium_change_percent,
        medium_fine_change_percent=total.medium_fine_change_percent,
        accepted=accepted,
        reason=reason,
        component_details={name: _component_payload(value) for name, value in components.items()},
    )
