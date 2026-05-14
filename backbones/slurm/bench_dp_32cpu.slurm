#!/bin/bash
#SBATCH --job-name=cs-bench-dp-32cpu
#SBATCH --output=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%j.out
#SBATCH --error=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%j.err
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=n-210
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:15:00

REPO=/home/morg/students/dortirosh/audio_ml_tau_final
VENV=/home/morg/students/dortirosh/envs/cs_amd
export PATH=$VENV/bin:${PATH:-}
export PYTHONPATH=$REPO:${PYTHONPATH:-}

# Honor the slurm allocation: most BLAS libs spawn their own thread pool.
# joblib processes will multiply that. Force per-process thread count to 1
# so per-item work stays single-threaded and joblib's n_jobs is the only knob.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$REPO"
echo "[bench] allocated CPUs: ${SLURM_CPUS_PER_TASK:-?}"
python /tmp/bench_dp.py --n-jobs 1 8 16 24 32 --repeats 5
