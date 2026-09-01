from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Sequence

from pipeline.models import CaseResult, GeometryResult, GridStudyResult, MeshProfile, MeshResult


_SCHEMA = """
CREATE TABLE IF NOT EXISTS designs (
    design_id INTEGER PRIMARY KEY,
    cad_sha256 TEXT NOT NULL UNIQUE,
    source_name TEXT NOT NULL,
    declared_unit TEXT NOT NULL,
    length_m REAL NOT NULL,
    width_m REAL NOT NULL,
    height_m REAL NOT NULL,
    volume_m3 REAL NOT NULL,
    wetted_area_m2 REAL NOT NULL,
    frontal_area_m2 REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    design_id INTEGER NOT NULL REFERENCES designs(design_id),
    revision INTEGER NOT NULL,
    config_sha256 TEXT NOT NULL,
    openfoam_version TEXT NOT NULL,
    openfoam_build TEXT NOT NULL,
    model_label TEXT NOT NULL,
    density_kg_m3 REAL NOT NULL,
    nu_m2_s REAL NOT NULL,
    temperature_c REAL NOT NULL,
    salinity_psu REAL NOT NULL,
    depth_m REAL NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    bow_axis TEXT NOT NULL,
    bow_confidence REAL NOT NULL,
    manifest_path TEXT NOT NULL,
    report_path TEXT,
    UNIQUE(design_id, config_sha256, openfoam_build, revision)
);

CREATE TABLE IF NOT EXISTS meshes (
    mesh_id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    profile TEXT NOT NULL,
    mesh_sha256 TEXT NOT NULL,
    cell_count INTEGER NOT NULL,
    surface_face_count INTEGER NOT NULL,
    layer_coverage_fraction REAL NOT NULL,
    max_non_orthogonality REAL NOT NULL,
    max_skewness REAL NOT NULL,
    diagnostic_min_wellposedness_determinant REAL NOT NULL,
    min_volume REAL NOT NULL,
    check_mesh_passed INTEGER NOT NULL CHECK (check_mesh_passed IN (0,1)),
    UNIQUE(run_id, profile)
);

CREATE TABLE IF NOT EXISTS cases (
    case_id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    mesh_id INTEGER NOT NULL REFERENCES meshes(mesh_id),
    result_key TEXT NOT NULL,
    revision INTEGER NOT NULL,
    speed_m_s REAL NOT NULL,
    mesh_profile TEXT NOT NULL,
    accepted INTEGER NOT NULL CHECK (accepted IN (0,1)),
    rejection_code TEXT,
    reynolds_number REAL NOT NULL,
    total_force_x_n REAL,
    total_force_y_n REAL,
    total_force_z_n REAL,
    pressure_force_x_n REAL,
    pressure_force_y_n REAL,
    pressure_force_z_n REAL,
    viscous_force_x_n REAL,
    viscous_force_y_n REAL,
    viscous_force_z_n REAL,
    drag_n REAL,
    pressure_drag_n REAL,
    viscous_drag_n REAL,
    side_force_n REAL,
    vertical_force_n REAL,
    roll_moment_nm REAL,
    pitch_moment_nm REAL,
    yaw_moment_nm REAL,
    drag_coefficient REAL,
    tow_power_w REAL,
    yplus_min REAL,
    yplus_p05 REAL,
    yplus_median REAL,
    yplus_mean REAL,
    yplus_p95 REAL,
    yplus_max REAL,
    yplus_fraction_below_30 REAL,
    yplus_fraction_30_to_300 REAL,
    yplus_fraction_above_300 REAL,
    cp_min REAL,
    minimum_absolute_pressure_pa REAL,
    cavitation_margin_pa REAL,
    force_stationary_mean_n REAL,
    force_stationary_cov REAL,
    force_mean_drift REAL,
    force_stationary_start_time_s REAL NOT NULL,
    force_stationary_end_time_s REAL NOT NULL,
    force_stationary_duration_flow_times REAL NOT NULL,
    force_mean_uncertainty_n REAL NOT NULL,
    force_mean_uncertainty_fraction REAL NOT NULL,
    force_integral_time_scale_s REAL NOT NULL,
    force_effective_sample_count REAL NOT NULL,
    residuals_json TEXT NOT NULL,
    physical_time_s REAL NOT NULL,
    time_steps INTEGER NOT NULL,
    maximum_courant_number REAL NOT NULL,
    UNIQUE(result_key, revision)
);

CREATE TABLE IF NOT EXISTS grid_studies (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
    reference_speed_m_s REAL NOT NULL,
    classification TEXT NOT NULL,
    r21 REAL NOT NULL,
    r32 REAL NOT NULL,
    observed_order REAL,
    extrapolated_drag_n REAL,
    fine_gci_percent REAL,
    accepted INTEGER NOT NULL CHECK (accepted IN (0,1)),
    reason TEXT NOT NULL,
    component_details_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS diagnostics (
    diagnostic_id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    stage TEXT NOT NULL,
    severity TEXT NOT NULL,
    code TEXT NOT NULL,
    message TEXT NOT NULL,
    artifact_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_cases_result_key ON cases(result_key);
CREATE INDEX IF NOT EXISTS idx_cases_run_speed ON cases(run_id, speed_m_s);
"""


SUMMARY_COLUMNS = (
    "run_id", "completed_at", "design_name", "cad_sha256", "config_sha256", "openfoam_version",
    "openfoam_build", "model_label", "bow_axis", "bow_confidence", "length_m", "width_m", "height_m",
    "volume_m3", "wetted_area_m2", "frontal_area_m2", "density_kg_m3", "nu_m2_s", "depth_m",
    "speed_m_s", "reynolds_number", "mesh_profile", "mesh_cells", "max_non_orthogonality",
    "max_skewness", "diagnostic_min_wellposedness_determinant", "layer_coverage_fraction", "yplus_min", "yplus_p05",
    "yplus_median", "yplus_mean", "yplus_p95", "yplus_max", "yplus_fraction_below_30",
    "yplus_fraction_30_to_300", "yplus_fraction_above_300", "total_force_x_n",
    "total_force_y_n", "total_force_z_n", "pressure_force_x_n", "pressure_force_y_n",
    "pressure_force_z_n", "viscous_force_x_n", "viscous_force_y_n", "viscous_force_z_n",
    "drag_n", "pressure_drag_n", "viscous_drag_n", "side_force_n", "vertical_force_n",
    "roll_moment_nm", "pitch_moment_nm", "yaw_moment_nm", "drag_coefficient", "tow_power_w",
    "cp_min", "minimum_absolute_pressure_pa", "cavitation_margin_pa", "force_stationary_cov",
    "force_mean_drift", "force_mean_uncertainty_fraction", "force_effective_sample_count",
    "force_stationary_duration_flow_times", "physical_time_s", "time_steps", "maximum_courant_number",
    "grid_classification", "reference_speed_gci_percent", "uncertainty_scope",
    "case_status", "report_path",
)


def _configuration(manifest: dict[str, object]) -> dict[str, object]:
    entry = manifest.get("configuration")
    if not isinstance(entry, dict) or not isinstance(entry.get("values"), dict):
        raise ValueError("manifest lacks the immutable configuration values required for persistence")
    return entry["values"]


def _nested(mapping: dict[str, object], key: str) -> dict[str, object]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"configuration section {key!r} is unavailable")
    return value


def _result_key(
    cad_sha256: str,
    config_sha256: str,
    openfoam_build: str,
    mesh_sha256: str,
    speed_m_s: float,
) -> str:
    payload = json.dumps(
        [cad_sha256, config_sha256, openfoam_build, mesh_sha256, f"{speed_m_s:.17g}"],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _started_at(run_id: str) -> str:
    timestamp = run_id.split("-", 1)[0]
    parsed = datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    return parsed.isoformat()


class ResultStore:
    def __init__(self, database: Path) -> None:
        self.database = database.resolve()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(_SCHEMA)

    def persist_run(
        self,
        manifest: dict[str, object],
        geometry: GeometryResult,
        meshes: Sequence[MeshResult],
        grid_study: GridStudyResult,
        cases: Sequence[CaseResult],
    ) -> None:
        self.initialize()
        run_id = str(manifest["run_root"]).rstrip("/").rsplit("/", 1)[-1]
        config_entry = manifest["configuration"]
        runtime = manifest["runtime"]
        if not isinstance(config_entry, dict) or not isinstance(runtime, dict):
            raise ValueError("manifest configuration/runtime metadata is invalid")
        config = _configuration(manifest)
        fluid = _nested(config, "fluid")
        operating = _nested(config, "operating")
        reports = manifest.get("reports")
        report_path = reports.get("html") if isinstance(reports, dict) else None
        completed_at = datetime.now(UTC).isoformat()
        extents = geometry.normalized_extents_m
        mesh_by_profile = {mesh.profile: mesh for mesh in meshes}
        if set(mesh_by_profile) != set(MeshProfile):
            raise ValueError("persistence requires coarse, medium, and fine meshes")
        if not cases:
            raise ValueError("persistence requires CFD case results")
        production_speeds = {float(value) for value in operating["speeds_m_s"]}  # type: ignore[index]
        medium_accepted = {
            case.speed_m_s for case in cases if case.mesh_profile == MeshProfile.MEDIUM and case.accepted
        }
        completed = grid_study.accepted and medium_accepted == production_speeds
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT run_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if existing is not None:
                connection.rollback()
                return
            connection.execute(
                """
                INSERT INTO designs (
                    cad_sha256, source_name, declared_unit, length_m, width_m, height_m,
                    volume_m3, wetted_area_m2, frontal_area_m2
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cad_sha256) DO UPDATE SET
                    source_name=excluded.source_name,
                    declared_unit=excluded.declared_unit,
                    length_m=excluded.length_m,
                    width_m=excluded.width_m,
                    height_m=excluded.height_m,
                    volume_m3=excluded.volume_m3,
                    wetted_area_m2=excluded.wetted_area_m2,
                    frontal_area_m2=excluded.frontal_area_m2
                """,
                (
                    geometry.source_sha256,
                    geometry.source.name,
                    geometry.declared_unit,
                    geometry.length_m,
                    extents[1],
                    extents[2],
                    geometry.volume_m3,
                    geometry.wetted_area_m2,
                    geometry.frontal_area_m2,
                ),
            )
            design_id = int(
                connection.execute("SELECT design_id FROM designs WHERE cad_sha256 = ?", (geometry.source_sha256,)).fetchone()[0]
            )
            revision = int(
                connection.execute(
                    "SELECT COALESCE(MAX(revision), -1) + 1 FROM runs WHERE design_id = ? AND config_sha256 = ? AND openfoam_build = ?",
                    (design_id, str(config_entry["digest"]), str(runtime["build"])),
                ).fetchone()[0]
            )
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, design_id, revision, config_sha256, openfoam_version, openfoam_build,
                    model_label, density_kg_m3, nu_m2_s, temperature_c, salinity_psu, depth_m,
                    started_at, completed_at, status, bow_axis, bow_confidence, manifest_path, report_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    design_id,
                    revision,
                    str(config_entry["digest"]),
                    str(runtime["version"]),
                    str(runtime["build"]),
                    str(config["model_label"]),
                    float(fluid["density_kg_m3"]),
                    float(fluid["kinematic_viscosity_m2_s"]),
                    float(fluid["temperature_c"]),
                    float(fluid["salinity_psu"]),
                    float(operating["centerline_depth_m"]),
                    _started_at(run_id),
                    completed_at,
                    "completed" if completed else "rejected",
                    geometry.bow_original_axis,
                    geometry.bow_confidence,
                    str(manifest["run_root"]) + "/manifest.json",
                    None if report_path is None else str(report_path),
                ),
            )
            mesh_ids: dict[MeshProfile, int] = {}
            for mesh in meshes:
                cursor = connection.execute(
                    """
                    INSERT INTO meshes (
                        run_id, profile, mesh_sha256, cell_count, surface_face_count,
                        layer_coverage_fraction, max_non_orthogonality, max_skewness,
                        diagnostic_min_wellposedness_determinant, min_volume, check_mesh_passed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        mesh.profile.value,
                        mesh.mesh_sha256,
                        mesh.cell_count,
                        mesh.surface_face_count,
                        mesh.layer_coverage_fraction,
                        mesh.max_non_orthogonality,
                        mesh.max_skewness,
                        mesh.diagnostic_min_wellposedness_determinant,
                        mesh.min_volume,
                        int(mesh.check_mesh_passed and mesh.mesh_ok),
                    ),
                )
                mesh_ids[mesh.profile] = int(cursor.lastrowid)
            for case in cases:
                yplus = case.yplus
                result_key = _result_key(
                    geometry.source_sha256,
                    str(config_entry["digest"]),
                    str(runtime["build"]),
                    case.mesh_sha256,
                    case.speed_m_s,
                )
                connection.execute(
                    """
                    INSERT INTO cases (
                        run_id, mesh_id, result_key, revision, speed_m_s, mesh_profile, accepted,
                        rejection_code, reynolds_number, total_force_x_n, total_force_y_n, total_force_z_n,
                        pressure_force_x_n, pressure_force_y_n, pressure_force_z_n, viscous_force_x_n,
                        viscous_force_y_n, viscous_force_z_n, drag_n, pressure_drag_n, viscous_drag_n,
                        side_force_n, vertical_force_n, roll_moment_nm, pitch_moment_nm, yaw_moment_nm,
                        drag_coefficient, tow_power_w, yplus_min, yplus_p05, yplus_median, yplus_mean,
                        yplus_p95, yplus_max, yplus_fraction_below_30, yplus_fraction_30_to_300,
                        yplus_fraction_above_300, cp_min, minimum_absolute_pressure_pa,
                        cavitation_margin_pa, force_stationary_mean_n, force_stationary_cov, force_mean_drift,
                        force_stationary_start_time_s, force_stationary_end_time_s,
                        force_stationary_duration_flow_times, force_mean_uncertainty_n,
                        force_mean_uncertainty_fraction, force_integral_time_scale_s,
                        force_effective_sample_count, residuals_json, physical_time_s,
                        time_steps, maximum_courant_number
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        run_id, mesh_ids[case.mesh_profile], result_key, revision, case.speed_m_s,
                        case.mesh_profile.value, int(case.accepted), case.rejection_code, case.reynolds_number,
                        *case.force_total_n, *case.force_pressure_n, *case.force_viscous_n, case.drag_n,
                        case.force_pressure_n[0], case.force_viscous_n[0], case.force_total_n[1],
                        case.force_total_n[2], *case.moment_nm, case.drag_coefficient, case.tow_power_w,
                        yplus["min"], yplus["p05"], yplus["median"], yplus["mean"], yplus["p95"],
                        yplus["max"], yplus["fraction_below_30"], yplus["fraction_30_to_300"],
                        yplus["fraction_above_300"], case.cp_min, case.minimum_absolute_pressure_pa,
                        case.cavitation_margin_pa, case.force_stationary_mean_n, case.force_stationary_cov,
                        case.force_mean_drift, case.force_stationary_start_time_s,
                        case.force_stationary_end_time_s, case.force_stationary_duration_flow_times,
                        case.force_mean_uncertainty_n, case.force_mean_uncertainty_fraction,
                        case.force_integral_time_scale_s, case.force_effective_sample_count,
                        json.dumps(case.residuals, sort_keys=True), case.physical_time_s,
                        case.time_steps, case.maximum_courant_number,
                    ),
                )
                for code in case.diagnostic_codes:
                    connection.execute(
                        "INSERT INTO diagnostics (run_id, stage, severity, code, message, artifact_path) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            run_id,
                            "solver",
                            "warning" if code.startswith("warning_") else "diagnostic",
                            code,
                            code.replace("_", " "),
                            str(case.case_dir / "case-result.json"),
                        ),
                    )
            connection.execute(
                """
                INSERT INTO grid_studies (
                    run_id, reference_speed_m_s, classification, r21, r32, observed_order,
                    extrapolated_drag_n, fine_gci_percent, accepted, reason, component_details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    float(operating["reference_speed_m_s"]),
                    grid_study.classification,
                    grid_study.effective_r21,
                    grid_study.effective_r32,
                    grid_study.observed_order,
                    grid_study.extrapolated_drag_n,
                    grid_study.fine_gci_percent,
                    int(grid_study.accepted),
                    grid_study.reason,
                    json.dumps(grid_study.component_details, sort_keys=True),
                ),
            )
            connection.commit()

    def summary_rows(self) -> list[dict[str, object]]:
        self.initialize()
        query = """
        WITH ranked AS (
            SELECT
                r.run_id, r.completed_at, d.source_name AS design_name, d.cad_sha256,
                r.config_sha256, r.openfoam_version, r.openfoam_build, r.model_label,
                r.bow_axis, r.bow_confidence, d.length_m, d.width_m, d.height_m,
                d.volume_m3, d.wetted_area_m2, d.frontal_area_m2, r.density_kg_m3,
                r.nu_m2_s, r.depth_m, c.speed_m_s, c.reynolds_number, c.mesh_profile,
                m.cell_count AS mesh_cells, m.max_non_orthogonality, m.max_skewness,
                m.diagnostic_min_wellposedness_determinant, m.layer_coverage_fraction, c.yplus_min, c.yplus_p05,
                c.yplus_median, c.yplus_mean, c.yplus_p95, c.yplus_max,
                c.yplus_fraction_below_30, c.yplus_fraction_30_to_300,
                c.yplus_fraction_above_300, c.total_force_x_n, c.total_force_y_n,
                c.total_force_z_n, c.pressure_force_x_n, c.pressure_force_y_n,
                c.pressure_force_z_n, c.viscous_force_x_n, c.viscous_force_y_n,
                c.viscous_force_z_n, c.drag_n, c.pressure_drag_n, c.viscous_drag_n,
                c.side_force_n, c.vertical_force_n, c.roll_moment_nm, c.pitch_moment_nm,
                c.yaw_moment_nm, c.drag_coefficient, c.tow_power_w, c.cp_min,
                c.minimum_absolute_pressure_pa, c.cavitation_margin_pa, c.force_stationary_cov,
                c.force_mean_drift, c.force_mean_uncertainty_fraction,
                c.force_effective_sample_count, c.force_stationary_duration_flow_times,
                c.physical_time_s, c.time_steps, c.maximum_courant_number,
                g.classification AS grid_classification,
                g.fine_gci_percent AS reference_speed_gci_percent,
                'reference_speed_only' AS uncertainty_scope,
                CASE WHEN c.accepted = 1 THEN 'accepted' ELSE 'rejected' END AS case_status,
                r.report_path,
                ROW_NUMBER() OVER (PARTITION BY c.result_key ORDER BY c.revision DESC) AS row_number
            FROM cases c
            JOIN runs r ON r.run_id = c.run_id
            JOIN designs d ON d.design_id = r.design_id
            JOIN meshes m ON m.mesh_id = c.mesh_id
            JOIN grid_studies g ON g.run_id = r.run_id
            WHERE r.status = 'completed' AND g.accepted = 1 AND c.accepted = 1 AND c.mesh_profile = 'medium'
        )
        SELECT * FROM ranked WHERE row_number = 1 ORDER BY design_name, speed_m_s
        """
        with self._connect() as connection:
            rows = connection.execute(query).fetchall()
        return [{column: row[column] for column in SUMMARY_COLUMNS} for row in rows]

    def export_summary_csv(self, destination: Path) -> None:
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = self.summary_rows()
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            key: format(value, ".17g") if isinstance(value, float) else value
                            for key, value in row.items()
                        }
                    )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def set_report_path(self, run_id: str, report_path: Path) -> None:
        self.initialize()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE runs SET report_path = ? WHERE run_id = ?",
                (str(report_path.resolve(strict=True)), run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"cannot attach report to unknown run: {run_id}")

    def table_counts(self) -> dict[str, int]:
        self.initialize()
        with self._connect() as connection:
            return {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("designs", "runs", "meshes", "cases", "grid_studies", "diagnostics")
            }

    def integrity_check(self) -> str:
        self.initialize()
        with self._connect() as connection:
            return str(connection.execute("PRAGMA integrity_check").fetchone()[0])

    def checkpoint(self) -> None:
        self.initialize()
        with self._connect() as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
