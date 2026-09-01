from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence

from pipeline.models import OpenFoamRuntime


def require_executable(name: str) -> Path:
    located = shutil.which(name)
    if located is None:
        raise RuntimeError(f"required executable not found on PATH: {name}")
    path = Path(located).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"required executable is not executable: {path}")
    return path


def discover_openfoam(env: Mapping[str, str] | None = None) -> OpenFoamRuntime:
    environment = os.environ if env is None else env
    version = environment.get("WM_PROJECT_VERSION", "")
    project_dir = Path(environment.get("WM_PROJECT_DIR", ""))
    expected_project_dir = Path("/usr/lib/openfoam/openfoam2512")
    if version != "2512" or project_dir != expected_project_dir:
        raise RuntimeError(f"OpenFOAM 2512 required; found version={version!r} dir={project_dir}")
    pimple_foam = require_executable("pimpleFoam")
    completed = subprocess.run([pimple_foam, "-help"], text=True, capture_output=True, check=True)
    build_line = next(
        (line for line in completed.stdout.splitlines() if line.startswith("Build:")),
        None,
    )
    if build_line is None:
        raise RuntimeError("pimpleFoam -help did not report an OpenFOAM build identifier")
    return OpenFoamRuntime(
        bashrc=Path("/usr/lib/openfoam/openfoam2512/etc/bashrc"),
        project_dir=project_dir,
        version=version,
        build=build_line.removeprefix("Build:").strip(),
        mpirun=require_executable("mpirun"),
        pimple_foam=pimple_foam,
    )


@dataclass(frozen=True)
class CommandFailure(RuntimeError):
    argv: tuple[str, ...]
    returncode: int
    log_path: Path

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, f"command failed with exit {self.returncode}; log: {self.log_path}")


def run_checked(
    argv: Sequence[str | Path],
    cwd: Path,
    log_path: Path,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = tuple(str(value) for value in argv)
    if not command:
        raise ValueError("command argv must not be empty")
    working_directory = cwd.resolve()
    if not working_directory.is_dir():
        raise ValueError(f"command working directory does not exist: {working_directory}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat()
    environment = None if env is None else {str(key): str(value) for key, value in env.items()}
    output: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"started_at={started_at}\n")
        log.write(f"cwd={working_directory}\n")
        log.write(f"argv={command!r}\n\n")
        log.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=working_directory,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            completed_at = datetime.now(UTC).isoformat()
            log.write(f"launch_error={error}\nended_at={completed_at}\nexit_status=127\n")
            log.flush()
            os.fsync(log.fileno())
            raise CommandFailure(command, 127, log_path) from error
        assert process.stdout is not None
        for line in process.stdout:
            output.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        returncode = process.wait()
        completed_at = datetime.now(UTC).isoformat()
        log.write(f"\nended_at={completed_at}\nexit_status={returncode}\n")
        log.flush()
        os.fsync(log.fileno())
    completed = subprocess.CompletedProcess(command, returncode, "".join(output), None)
    if returncode != 0:
        raise CommandFailure(command, returncode, log_path)
    return completed
