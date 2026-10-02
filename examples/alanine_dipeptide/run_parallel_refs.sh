#!/usr/bin/env bash
# A1 reference as N independent trajectories run in parallel (user decision
# 2026-10-01), instead of one long run. Each run: ref_long.py, its own seed and
# output dir, from the same equilibrated system (runs/ala2_system); the
# analysis discards the first 10 ns of every run (--skip-ps 10000, ~3 t2).
#
# Usage (foreground; Ctrl+C stops every run cleanly, rerun the same command
# to resume them):
#   bash examples/alanine_dipeptide/run_parallel_refs.sh GPUS [N] [NS] [CORES]
#     GPUS   comma-separated CUDA device ids, runs are spread round-robin
#            (ala2 does not saturate a GPU: all runs on one GPU are fine)
#     N      number of runs (default 8), NS ns per run (default 100)
#     CORES  comma-separated CPU ids, one per run, each run pinned to its own
#            (taskset). Default: one logical CPU per distinct physical core
#            from lscpu (no two runs on hyperthread siblings). The CUDA
#            platform busy-waits on one host thread, so every run keeps one
#            core at 100 %; sharing a core slows both runs.
#
# Then analyse together with the first run cut into 2 pieces:
#   python examples/alanine_dipeptide/analyze_ref.py --run runs/ala2_ref:2 \
#     $(for d in runs/ala2_par/r*/; do printf -- '--run %s ' "${d%/}"; done) \
#     --out runs/ala2_par/analysis --skip-ps 10000
set -euo pipefail
cd "$(dirname "$0")/../.."

IFS=, read -r -a gpus <<< "${1:?usage: $0 GPUS [N] [NS] [CORES]}"
n=${2:-8}
ns=${3:-100}
if [[ -n ${4:-} ]]; then
  IFS=, read -r -a cores <<< "$4"
else
  # first logical CPU of each (socket, core), in order
  mapfile -t cores < <(lscpu -p=CPU,CORE,SOCKET | grep -v '^#' | sort -t, -k3,3n -k2,2n -k1,1n \
                       | awk -F, '!seen[$3","$2]++ {print $1}')
fi
if (( ${#cores[@]} < n )); then
  echo "only ${#cores[@]} cores for $n runs; pass CORES explicitly or fewer runs" >&2
  exit 1
fi
root=runs/ala2_par
mkdir -p "$root"
# one host thread per run: no extra BLAS/OpenMP threads competing for the cores
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

pids=()
trap 'kill -TERM "${pids[@]}" 2>/dev/null || true; wait' INT TERM
for ((i = 1; i <= n; i++)); do
  d=$(printf '%s/r%02d' "$root" "$i")
  gpu=${gpus[$(((i - 1) % ${#gpus[@]}))]}
  core=${cores[$((i - 1))]}
  resume=()
  [[ -f $d/run.json ]] && resume=(--resume)
  CUDA_VISIBLE_DEVICES=$gpu taskset -c "$core" \
    python -u examples/alanine_dipeptide/ref_long.py \
      --out "$d" --total-ns "$ns" --seed $((20261001 + i)) --platform CUDA \
      --system-dir "$PWD/runs/ala2_system" "${resume[@]}" \
      >> "$d.log" 2>&1 &
  pids+=($!)
  echo "run $d: GPU $gpu, CPU $core, pid $!"
done
echo "progress: tail -n1 $root/r*/progress.log"
wait
