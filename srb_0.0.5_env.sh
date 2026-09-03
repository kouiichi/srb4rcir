#!/usr/bin/env bash
set -euo pipefail

export SRB_ROOT="/home/ubuntu/yyf/space_robotics_bench_0.0.5"
export ISAAC_SIM_PATH="/home/ubuntu/isaac-sim-4.5"
export ISAAC_SIM_PYTHON="${ISAAC_SIM_PATH}/python.sh"
export ISAAC_ML_PREBUNDLE="${ISAAC_SIM_PATH}/exts/omni.isaac.ml_archive/pip_prebundle"
export BLENDER_PATH="/home/ubuntu/blender-4.3.2-linux-x64"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/srb_0.0.5_matplotlib}"

if [[ ! -x "${ISAAC_SIM_PYTHON}" ]]; then
    echo "[ERROR] ISAAC_SIM_PYTHON is not executable: ${ISAAC_SIM_PYTHON}" >&2
    exit 1
fi

if [[ ! -d "${SRB_ROOT}" ]]; then
    echo "[ERROR] SRB_ROOT does not exist: ${SRB_ROOT}" >&2
    exit 1
fi

_prepend_path() {
    local var_name="$1"
    local path_value="$2"
    if [[ -d "${path_value}" ]]; then
        eval "local current_value=\"\${${var_name}:-}\""
        case ":${current_value}:" in
            *":${path_value}:"*) ;;
            *) export "${var_name}=${path_value}${current_value:+:${current_value}}" ;;
        esac
    fi
}

_prepend_path PATH "${ISAAC_SIM_PATH}/kit/python/bin"
_prepend_path PATH "${BLENDER_PATH}"
mkdir -p "${MPLCONFIGDIR}"

for lib_dir in \
    "${ISAAC_ML_PREBUNDLE}/nvidia/cuda_cupti/lib" \
    "${ISAAC_ML_PREBUNDLE}/nvidia/cuda_runtime/lib" \
    "${ISAAC_ML_PREBUNDLE}/nvidia/cublas/lib" \
    "${ISAAC_ML_PREBUNDLE}/nvidia/cudnn/lib" \
    "${ISAAC_ML_PREBUNDLE}/nvidia/cufft/lib" \
    "${ISAAC_ML_PREBUNDLE}/nvidia/curand/lib" \
    "${ISAAC_ML_PREBUNDLE}/nvidia/cusolver/lib" \
    "${ISAAC_ML_PREBUNDLE}/nvidia/cusparse/lib" \
    "${ISAAC_ML_PREBUNDLE}/nvidia/nccl/lib" \
    "${ISAAC_ML_PREBUNDLE}/nvidia/nvtx/lib"
do
    _prepend_path LD_LIBRARY_PATH "${lib_dir}"
done

cd "${SRB_ROOT}"

if [[ "$#" -eq 0 ]]; then
    cat <<EOF
[INFO] SRB 0.0.5 environment is ready.
[INFO] SRB_ROOT=${SRB_ROOT}
[INFO] ISAAC_SIM_PYTHON=${ISAAC_SIM_PYTHON}

Run a command through this wrapper, for example:
  ${SRB_ROOT}/srb_0.0.5_env.sh torch-check
  ${SRB_ROOT}/srb_0.0.5_env.sh srb-ls
  ${SRB_ROOT}/srb_0.0.5_env.sh train-rendezvous
  ${SRB_ROOT}/srb_0.0.5_env.sh -- python -c "import torch; print(torch.__version__)"

Or source it in the current shell:
  source ${SRB_ROOT}/srb_0.0.5_env.sh
EOF
    return 0 2>/dev/null || exit 0
fi

case "$1" in
    torch-check)
        exec "${ISAAC_SIM_PYTHON}" -c "import torch; print(torch.__file__); print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count())"
        ;;
    sim-check)
        exec "${ISAAC_SIM_PYTHON}" -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); import isaacsim.core.utils.stage as stage_utils; import isaaclab; print('isaac sim + isaaclab ok'); app.close()"
        ;;
    srb-help)
        exec "${ISAAC_SIM_PYTHON}" -m srb --help
        ;;
    srb-ls)
        exec "${ISAAC_SIM_PYTHON}" -m srb ls
        ;;
    srb)
        shift
        exec "${ISAAC_SIM_PYTHON}" -m srb "$@"
        ;;
    train-rendezvous)
        exec "${ISAAC_SIM_PYTHON}" -m srb agent train --headless --algo sb3_ppo -e mobile/rendezvous env.num_envs=4
        ;;
    --)
        shift
        if [[ "${1:-}" == "srb" ]]; then
            shift
            exec "${ISAAC_SIM_PYTHON}" -m srb "$@"
        fi
        exec "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
