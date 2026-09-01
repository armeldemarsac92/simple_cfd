#!/usr/bin/env bash
set -Eeuo pipefail
readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly VENV_DIR="${PROJECT_ROOT}/.venv"
export PIP_CACHE_DIR="${PROJECT_ROOT}/.cache/pip"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
mkdir -p "${PIP_CACHE_DIR}" "${MPLCONFIGDIR}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    python3 -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install --requirement "${PROJECT_ROOT}/requirements.txt"
"${VENV_DIR}/bin/python" -c 'import gmsh, jinja2, matplotlib, meshio, numpy, scipy, shapely, trimesh; print("Python CFD dependencies ready")'
