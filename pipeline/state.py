from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from pipeline.models import RunPaths, Stage, StageStatus


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    slug = slug.strip("-")
    if not slug:
        raise ValueError("value cannot produce an empty slug")
    return slug


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within(root: Path, candidate: Path) -> Path:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes run root: {candidate}") from error
    return resolved


def _status_payload(run_id: str) -> dict[str, Any]:
    timestamp = datetime.now(UTC).isoformat()
    return {
        "schema_version": 2,
        "run_id": run_id,
        "current_stage": None,
        "stages": {
            stage.value: {"status": StageStatus.PENDING.value, "updated_at": timestamp}
            for stage in Stage
        },
    }


def create_run_paths(project_root: Path, cad_sha: str, config_digest: str) -> RunPaths:
    if len(cad_sha) < 12 or len(config_digest) < 8:
        raise ValueError("CAD and configuration digests are too short for a run ID")
    root = project_root.resolve()
    runs_root = (root / "runs").resolve()
    _within(root, runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{cad_sha[:12]}-{config_digest[:8]}"
    run_root = _within(runs_root, runs_root / run_id)
    run_root.mkdir(mode=0o755, exist_ok=False)
    paths = RunPaths(
        root=run_root,
        manifest=_within(runs_root, run_root / "manifest.json"),
        status=_within(runs_root, run_root / "status.json"),
        geometry=_within(runs_root, run_root / "geometry"),
        meshes=_within(runs_root, run_root / "meshes"),
        cases=_within(runs_root, run_root / "cases"),
        postprocessing=_within(runs_root, run_root / "postprocessing"),
    )
    for directory in (paths.geometry, paths.meshes, paths.cases, paths.postprocessing):
        directory.mkdir(mode=0o755)
    atomic_write_json(paths.status, _status_payload(run_id))
    return paths


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def read_status(path: Path | RunPaths) -> dict[str, Any]:
    status_path = path.status if isinstance(path, RunPaths) else path
    with status_path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or value.get("schema_version") != 2 or not isinstance(value.get("stages"), dict):
        raise ValueError(f"invalid run status file: {status_path}")
    return value


def append_run_event(paths: RunPaths, value: Mapping[str, object]) -> Path:
    event_path = paths.root / "run-events.jsonl"
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": paths.root.name,
        **dict(value),
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n"
    with event_path.open("a", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    return event_path


_ALLOWED_TRANSITIONS = {
    StageStatus.PENDING: {StageStatus.RUNNING, StageStatus.REJECTED},
    StageStatus.RUNNING: {StageStatus.ACCEPTED, StageStatus.REJECTED},
    StageStatus.ACCEPTED: set(),
    StageStatus.REJECTED: set(),
}


def transition_stage(
    paths: RunPaths,
    stage: Stage,
    status: StageStatus,
    *,
    detail: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    state = read_status(paths)
    entry = state["stages"].get(stage.value)
    if not isinstance(entry, dict):
        raise ValueError(f"missing stage in run status: {stage.value}")
    try:
        previous = StageStatus(entry["status"])
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid status for stage {stage.value}") from error
    if status not in _ALLOWED_TRANSITIONS[previous]:
        raise ValueError(f"invalid {stage.value} transition: {previous.value} -> {status.value}")
    replacement: dict[str, object] = {"status": status.value, "updated_at": datetime.now(UTC).isoformat()}
    if detail is not None:
        replacement["detail"] = dict(detail)
    state["stages"][stage.value] = replacement
    state["run_id"] = paths.root.name
    state["current_stage"] = stage.value
    atomic_write_json(paths.status, state)
    append_run_event(
        paths,
        {
            "event": "stage_transition",
            "stage": stage.value,
            "previous_status": previous.value,
            "status": status.value,
            "detail": dict(detail) if detail is not None else {},
        },
    )
    return state
