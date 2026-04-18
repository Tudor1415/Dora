#!/bin/bash
#SBATCH --job-name=dora_dental_latent_sweep
#SBATCH -p prepost
#SBATCH -A wbw@v100
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Keep nounset disabled: Jean Zay conda activation scripts reference unset vars.
set -eo pipefail
set -x

# Ensure folder exists for any extra logs created during execution.
# For Slurm's own open(2) at job start, create it before sbatch:
#   mkdir -p slurm_logs
mkdir -p slurm_logs

module purge
module load python/3.9.12
module load cuda/12.8.0

VENV_ACTIVATE="${VENV_ACTIVATE:-$WORK/torch-env/bin/activate}"
if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "[ERROR] Python environment activate script not found: ${VENV_ACTIVATE}"
  exit 2
fi
source "${VENV_ACTIVATE}"
echo "[INFO] which python: $(which python)"
python --version

PROJECT_ROOT="${PROJECT_ROOT:-$WORK/Dora}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH/seg_data}"
PRECOMP_ROOT="${PRECOMP_ROOT:-$SCRATCH/seg_data_precomp_sdf}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/infer_out}"

# Dependency bootstrap:
# - default behavior installs only missing OmegaConf (fastest fix for current failure)
# - set INSTALL_REQUIREMENTS=1 to install full pytorch_lightning/requirements.txt
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"
AUTO_INSTALL_OMEGACONF="${AUTO_INSTALL_OMEGACONF:-1}"

if [[ "${INSTALL_REQUIREMENTS}" == "1" ]]; then
  echo "[INFO] Installing full Python requirements..."
  python -m pip install -r "$PROJECT_ROOT/pytorch_lightning/requirements.txt"
else
  if ! python - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("omegaconf") is not None else 1)
PY
  then
    if [[ "${AUTO_INSTALL_OMEGACONF}" == "1" ]]; then
      echo "[WARN] Missing dependency: omegaconf. Installing omegaconf==2.3.0..."
      python -m pip install "omegaconf==2.3.0"
    else
      echo "[ERROR] Missing dependency: omegaconf"
      echo "[ERROR] Re-run with AUTO_INSTALL_OMEGACONF=1 or install manually:"
      echo "        python -m pip install omegaconf==2.3.0"
      exit 2
    fi
  fi
fi

DEVICE="${DEVICE:-cuda}"
SCAN_SEED="${SCAN_SEED:-0}"
LATENT_LENGTHS="${LATENT_LENGTHS:-256,512,1024,2048,4096}"
SIGMAS="${SIGMAS:-0.01,0.03}"
LOCAL_SAMPLES_TOTAL="${LOCAL_SAMPLES_TOTAL:-6}"
N_SUPERVISION="${N_SUPERVISION:-21384,10000,10000}"
OCTREE_DEPTH="${OCTREE_DEPTH:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-262144}"
ANGLE_THRESHOLD="${ANGLE_THRESHOLD:-15}"
POINT_NUMBER="${POINT_NUMBER:-65536}"

CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/pytorch_lightning/configs/shape-autoencoder/Dora-VAE-test.yaml}"
DEFAULT_CKPT="$PROJECT_ROOT/pytorch_lightning/ckpts/Dora-VAE-1.1/dora_vae_1_1.ckpt"
CKPT="${CKPT:-}"
if [[ -z "${CKPT}" && -f "${DEFAULT_CKPT}" ]]; then
  CKPT="${DEFAULT_CKPT}"
fi

cd "$PROJECT_ROOT/pytorch_lightning"

CMD=(
  python infer_dental_latent_sweep.py
  --config "${CONFIG_PATH}"
  --data_root "${DATA_ROOT}"
  --precomp_root "${PRECOMP_ROOT}"
  --out_dir "${OUT_ROOT}"
  --device "${DEVICE}"
  --scan_seed "${SCAN_SEED}"
  --latent_lengths "${LATENT_LENGTHS}"
  --sigmas "${SIGMAS}"
  --local_samples_total "${LOCAL_SAMPLES_TOTAL}"
  --n_supervision "${N_SUPERVISION}"
  --octree_depth "${OCTREE_DEPTH}"
  --chunk_size "${CHUNK_SIZE}"
  --angle_threshold "${ANGLE_THRESHOLD}"
  --point_number "${POINT_NUMBER}"
)

if [[ -n "${CKPT}" ]]; then
  CMD+=(--ckpt "${CKPT}")
fi

if [[ "${FORCE_RESAMPLE:-0}" == "1" ]]; then
  CMD+=(--force_resample)
fi

if [[ "${SAMPLE_POSTERIOR:-0}" == "1" ]]; then
  CMD+=(--sample_posterior)
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] PRECOMP_ROOT=${PRECOMP_ROOT}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] CKPT=${CKPT:-<from config>}"
echo "[INFO] Command: ${CMD[*]}"
"${CMD[@]}"
