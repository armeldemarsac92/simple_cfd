from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from pipeline.models import CaseResult, GeometryResult, GridStudyResult, MeshProfile, MeshResult, RunPaths
from pipeline.postprocess import parse_force_history, parse_solver_info, read_wall_surface
from pipeline.store import ResultStore
from pipeline.visualization import render_case_visuals


_TEMPLATES = Path(__file__).resolve().parent / "templates"


def _atomic_text(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path.resolve(strict=True)


def _save_figure(figure: plt.Figure, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    figure.savefig(temporary, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)
    return destination.resolve(strict=True)


def _relative(path: Path | None, base: Path) -> str:
    if path is None:
        return ""
    return os.path.relpath(path.resolve(), base.resolve()).replace(os.sep, "/")


def _case_diagnostics(case: CaseResult, destination: Path) -> Path:
    force = parse_force_history(case.force_history_path)
    residuals = parse_solver_info(case.solver_info_path)
    surface = read_wall_surface(case.wall_fields_path)
    yplus_names = [name for name in surface.cell_fields if name == "yPlus" or name.rsplit(":", 1)[-1] == "yPlus"]
    if len(yplus_names) != 1:
        raise ValueError(f"expected one yPlus field in {case.wall_fields_path}")
    yplus = np.asarray(surface.cell_fields[yplus_names[0]], dtype=float).reshape(-1)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    axes[0].plot(force.times_s, force.total_force_n[:, 0], label="total", linewidth=1.2)
    axes[0].plot(force.times_s, force.pressure_force_n[:, 0], label="pressure", linewidth=1.0)
    axes[0].plot(force.times_s, force.viscous_force_n[:, 0], label="viscous", linewidth=1.0)
    axes[0].axvspan(
        case.force_stationary_start_time_s,
        case.force_stationary_end_time_s,
        alpha=0.12,
        color="#2f855a",
        label="stationary sample",
    )
    stationary = (force.times_s >= case.force_stationary_start_time_s) & (
        force.times_s <= case.force_stationary_end_time_s
    )
    stationary_times = force.times_s[stationary]
    stationary_drag = force.total_force_n[stationary, 0]
    if len(stationary_times) >= 2:
        increments = np.diff(stationary_times)
        integral = np.cumsum(0.5 * (stationary_drag[1:] + stationary_drag[:-1]) * increments)
        running_mean = integral / (stationary_times[1:] - stationary_times[0])
        axes[0].plot(stationary_times[1:], running_mean, label="running mean", linewidth=1.5, color="#22543d")
    axes[0].set(xlabel="Physical time (s)", ylabel="X force (N)", title="Drag and converged mean")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    for field, history in residuals.items():
        axes[1].semilogy(history.times_s, np.maximum(history.initial, 1.0e-30), label=field, linewidth=0.9)
    axes[1].set(xlabel="Physical time (s)", ylabel="Initial residual", title="PIMPLE per-step residuals")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=7, ncol=2)
    positive = yplus[np.isfinite(yplus) & (yplus > 0.0)]
    axes[2].hist(positive, bins=np.geomspace(max(float(np.min(positive)), 0.1), float(np.max(positive)) * 1.001, 55), color="#3182ce", alpha=0.82)
    axes[2].axvspan(30.0, 300.0, color="#48bb78", alpha=0.15, label="30–300")
    axes[2].set_xscale("log")
    axes[2].set(xlabel="y+", ylabel="Hull faces", title="Wall-function distribution")
    axes[2].grid(alpha=0.25, which="both")
    axes[2].legend(fontsize=8)
    figure.suptitle(f"{case.mesh_profile.value} grid — U∞={case.speed_m_s:.2f} m/s", fontsize=14)
    return _save_figure(figure, destination)


def _mesh_grid_plot(meshes: Sequence[MeshResult], cases: Sequence[CaseResult], grid: GridStudyResult, destination: Path) -> Path:
    ordered_meshes = sorted(meshes, key=lambda item: item.cell_count)
    reference = {case.mesh_profile: case for case in cases if abs(case.speed_m_s - 1.0) < 1.0e-12}
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    labels = [mesh.profile.value for mesh in ordered_meshes]
    cells = [mesh.cell_count for mesh in ordered_meshes]
    colors = ["#90cdf4", "#4299e1", "#2b6cb0"]
    axes[0].bar(labels, cells, color=colors)
    for index, count in enumerate(cells):
        axes[0].text(index, count, f"{count:,}", ha="center", va="bottom", fontsize=9)
    axes[0].set(ylabel="Finite-volume cells", title="Three-grid family")
    axes[0].grid(axis="y", alpha=0.25)
    profiles = [MeshProfile.COARSE, MeshProfile.MEDIUM, MeshProfile.FINE]
    drag = [reference[profile].drag_n for profile in profiles]
    axes[1].plot([1, 2, 3], drag, marker="o", linewidth=1.8, color="#2b6cb0")
    if grid.extrapolated_drag_n is not None:
        axes[1].axhline(grid.extrapolated_drag_n, linestyle="--", color="#c53030", label="Richardson extrapolation")
        axes[1].legend(fontsize=8)
    axes[1].set(xticks=[1, 2, 3], xticklabels=[profile.value for profile in profiles], ylabel="Drag at 1.00 m/s (N)", title=f"{grid.classification}; GCI={grid.fine_gci_percent}%")
    axes[1].grid(alpha=0.25)
    return _save_figure(figure, destination)


def _sweep_plot(cases: Sequence[CaseResult], destination: Path) -> Path:
    ordered = sorted((case for case in cases if case.mesh_profile == MeshProfile.MEDIUM), key=lambda item: item.speed_m_s)
    speed = np.asarray([case.speed_m_s for case in ordered])
    total = np.asarray([case.drag_n for case in ordered])
    pressure = np.asarray([case.force_pressure_n[0] for case in ordered])
    viscous = np.asarray([case.force_viscous_n[0] for case in ordered])
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    axes[0].plot(speed, total, "o-", label="total")
    axes[0].plot(speed, pressure, "o-", label="pressure")
    axes[0].plot(speed, viscous, "o-", label="viscous")
    axes[0].set(xlabel="Speed (m/s)", ylabel="Drag (N)", title="Drag decomposition")
    axes[0].legend(fontsize=8)
    axes[1].plot([case.reynolds_number for case in ordered], [case.drag_coefficient for case in ordered], "o-", color="#805ad5")
    axes[1].set(xlabel="Reynolds number", ylabel="Cd", title="Drag coefficient")
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[2].plot(speed, [case.tow_power_w for case in ordered], "o-", color="#c05621")
    axes[2].set(xlabel="Speed (m/s)", ylabel="Tow power (W)", title="Ideal tow power")
    for axis in axes:
        axis.grid(alpha=0.25)
    return _save_figure(figure, destination)


def _case_artifact_context(
    case: CaseResult,
    geometry: GeometryResult,
    report_dir: Path,
) -> tuple[dict[str, object], tuple[str, ...]]:
    destination = case.case_dir / "visualizations"
    warnings: list[str] = []
    visual = None
    try:
        visual = render_case_visuals(case, geometry, destination)
        warnings.extend(visual.warnings)
    except (OSError, RuntimeError, ValueError) as error:
        warnings.append(f"visualization_failed: {error}")
    diagnostics_plot: Path | None = None
    try:
        diagnostics_plot = _case_diagnostics(case, destination / "numerical-history.png")
    except (OSError, RuntimeError, ValueError) as error:
        warnings.append(f"diagnostic_plot_failed: {error}")
    return (
        {
            "case": case,
            "diagnostics_plot": _relative(diagnostics_plot, report_dir),
            "case_marker": _relative(None if visual is None else visual.case_marker, report_dir),
            "hull_vtk": _relative(None if visual is None else visual.hull_vtk, report_dir),
            "plane_vtk": _relative(None if visual is None else visual.plane_vtk, report_dir),
            "cp_png": _relative(None if visual is None else visual.cp_png, report_dir),
            "wall_shear_png": _relative(None if visual is None else visual.wall_shear_png, report_dir),
            "yplus_png": _relative(None if visual is None else visual.yplus_png, report_dir),
            "flow_png": _relative(None if visual is None else visual.flow_png, report_dir),
            "warnings": tuple(warnings),
        },
        tuple(warnings),
    )


def render_run_report(
    paths: RunPaths,
    manifest: dict[str, object],
    geometry: GeometryResult,
    meshes: Sequence[MeshResult],
    grid_study: GridStudyResult,
    cases: Sequence[CaseResult],
) -> tuple[Path, Path]:
    report_dir = paths.root / "reports"
    figure_dir = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    production = sorted(
        (case for case in cases if case.mesh_profile == MeshProfile.MEDIUM),
        key=lambda item: item.speed_m_s,
    )
    mesh_grid = _mesh_grid_plot(meshes, cases, grid_study, figure_dir / "mesh-grid-qualification.png")
    sweep = _sweep_plot(production, figure_dir / "speed-sweep.png")
    artifact_context: list[dict[str, object]] = []
    report_warnings: list[str] = []
    for case in sorted(cases, key=lambda item: (item.speed_m_s, item.mesh_profile.value)):
        if not case.accepted:
            continue
        item, warnings = _case_artifact_context(case, geometry, report_dir)
        artifact_context.append(item)
        report_warnings.extend(warnings)
    config_entry = manifest.get("configuration")
    runtime = manifest.get("runtime")
    if not isinstance(config_entry, dict) or not isinstance(config_entry.get("values"), dict) or not isinstance(runtime, dict):
        raise ValueError("report requires immutable configuration and runtime metadata")
    config = config_entry["values"]
    fluid = config["fluid"]
    operating = config["operating"]
    environment = Environment(loader=FileSystemLoader(_TEMPLATES), undefined=StrictUndefined, autoescape=False)
    context = {
        "run_id": paths.root.name,
        "status": "accepted" if grid_study.accepted and all(case.accepted for case in production) else "rejected",
        "geometry": geometry,
        "meshes": sorted(meshes, key=lambda item: item.cell_count),
        "grid": grid_study,
        "production_cases": production,
        "case_artifacts": artifact_context,
        "report_warnings": tuple(report_warnings),
        "model_label": config["model_label"],
        "density": float(fluid["density_kg_m3"]),
        "nu": float(fluid["kinematic_viscosity_m2_s"]),
        "depth_m": float(operating["centerline_depth_m"]),
        "runtime": runtime,
        "configuration": config_entry,
        "manifest_path": _relative(paths.manifest, report_dir),
        "grid_path": _relative(paths.postprocessing / "grid-study.json", report_dir),
        "geometry_preview": _relative(geometry.preview_png, report_dir),
        "summary_plots": {"mesh_grid": _relative(mesh_grid, report_dir), "sweep": _relative(sweep, report_dir)},
    }
    markdown = _atomic_text(report_dir / "report.md", environment.get_template("report.md.j2").render(**context))
    html = _atomic_text(report_dir / "report.html", environment.get_template("report.html.j2").render(**context))
    return markdown, html


def render_comparison_report(store: ResultStore, destination: Path) -> Path:
    rows = store.summary_rows()
    figure, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["design_name"]), str(row["cad_sha256"])), []).append(row)
    for (name, digest), values in grouped.items():
        ordered = sorted(values, key=lambda item: float(item["speed_m_s"]))
        axis.plot(
            [float(item["speed_m_s"]) for item in ordered],
            [float(item["drag_n"]) for item in ordered],
            marker="o",
            label=f"{name} ({digest[:12]})",
        )
    axis.set(xlabel="Speed (m/s)", ylabel="Drag (N)", title="Accepted AUV hull comparison")
    axis.grid(alpha=0.25)
    if grouped:
        axis.legend(fontsize=8)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    chart = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii") if rows else ""
    environment = Environment(loader=FileSystemLoader(_TEMPLATES), undefined=StrictUndefined, autoescape=True)
    rendered = environment.get_template("compare.html.j2").render(
        rows=rows,
        columns=("design_name", "cad_sha256", "speed_m_s", "drag_n", "drag_coefficient", "tow_power_w", "reference_speed_gci_percent", "report_path"),
        chart_data_uri=chart,
    )
    return _atomic_text(destination.resolve(), rendered)
