#!/bin/bash
# Template for running a batch of simulations on a SLURM cluster.
#
# Adjust the #SBATCH directives, the environment activation, and the paths below
# for your site, then submit with:
#
#   sbatch examples/run_slurm.sh
#
# The job is single-node and CPU-only; no GPU is required.

#SBATCH --job-name=sptrace-batch
#SBATCH --output=logs/sptrace-batch-%j.out
#SBATCH --error=logs/sptrace-batch-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

# ---- configure ------------------------------------------------------------
# Repository root. Defaults to the parent of this script's directory.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Design file and output directory.
DESIGN="${DESIGN:-$REPO_ROOT/configs/design.toml}"
OUTPUT="${OUTPUT:-$REPO_ROOT/out/batch}"

# Optional: genetic map for the Numbat export. Leave empty to skip that step.
# Ships with Eagle v2.4.1 as tables/genetic_map_hg38_withX.txt.gz.
GENETIC_MAP="${GENETIC_MAP:-}"

# Optional: environment to activate. Delete if the package is already on PATH.
CONDA_ENV="${CONDA_ENV:-}"
# ---------------------------------------------------------------------------

mkdir -p "$(dirname "$OUTPUT")" logs

if [[ -n "$CONDA_ENV" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

echo "Job ID   : ${SLURM_JOB_ID:-<interactive>}"
echo "Node     : ${SLURMD_NODENAME:-$(hostname)}"
echo "Date     : $(date -Is)"
echo "Python   : $(command -v python)"
echo "Version  : $(python -c 'import sptrace as s; print(s.__version__)')"
echo "Design   : $DESIGN"
echo "Output   : $OUTPUT"

cd "$REPO_ROOT"

args=(--design "$DESIGN" --output "$OUTPUT")
if [[ -n "$GENETIC_MAP" ]]; then
    args+=(--genetic-map "$GENETIC_MAP")
else
    echo "NOTE: GENETIC_MAP not set; skipping the Numbat export."
fi

# Replicates are memory-hungry; they run sequentially, each in its own process.
# --jobs 1 keeps expression generation reproducible (and is not slower).
sptrace-batch "${args[@]}" --jobs 1 2>&1 | tee "$OUTPUT/batch_log.txt"

echo "Finished at: $(date -Is)"