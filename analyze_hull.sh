#!/usr/bin/env bash
set -Eeuo pipefail
readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly OPENFOAM_ENV="${OPENFOAM_BASHRC:-/usr/lib/openfoam/openfoam2512/etc/bashrc}"
if [[ ! -r "${OPENFOAM_ENV}" ]]; then
    printf 'OpenFOAM v2512 bashrc not found: %s\n' "${OPENFOAM_ENV}" >&2
    exit 2
fi
if [[ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    printf 'Local environment missing. Run ./setup.sh once.\n' >&2
    exit 2
fi
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
set +eu
source "${OPENFOAM_ENV}"
openfoam_source_status=$?
set -eu
if [[ ${openfoam_source_status} -ne 0 ]]; then
    printf 'OpenFOAM v2512 environment setup failed: %s\n' "${OPENFOAM_ENV}" >&2
    exit 2
fi
exec "${PROJECT_ROOT}/.venv/bin/python" -m pipeline.cli "$@"
