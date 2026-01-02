#!/usr/bin/env bash
#SBATCH --job-name=drone
#SBATCH --account=def-smaley
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8       # nominal "cores"; ORCA runs without MPI
#SBATCH --cpus-per-task=1  # threads for ORCA
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=ALL

set -euo pipefail

# ---------- Resolve jobs root (defaults to submit dir) ----------
JOBS_ROOT="${JOBS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if command -v realpath >/dev/null 2>&1; then JOBS_ROOT="$(realpath "$JOBS_ROOT")"; fi
echo "[worker] Using JOBS_ROOT=$JOBS_ROOT"

# ---------- ORCA module + binary ----------
module purge >/dev/null 2>&1 || true
module load orca/6.1.0
ORCA_BIN="${EBROOTORCA}/orca"
if [[ ! -x "$ORCA_BIN" ]]; then
  echo "[worker] ERROR: ORCA not found at ${EBROOTORCA}/orca" >&2
  exit 2
fi

# Threaded execution only (no MPI)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
ulimit -s unlimited

# ---------- Scratch ----------
SCR="${SLURM_TMPDIR:-/tmp}/${SLURM_JOB_NAME}_${SLURM_JOB_ID:-$$}"
mkdir -p "$SCR"
export ORCA_TMPDIR="$SCR"

# ---------- Helpers ----------
log_counts() {
  local rd st dn fl
  rd=$(find "$JOBS_ROOT" -mindepth 2 -maxdepth 2 -type f -name READY   | wc -l)
  st=$(find "$JOBS_ROOT" -mindepth 2 -maxdepth 2 -type f -name STARTED | wc -l)
  dn=$(find "$JOBS_ROOT" -mindepth 2 -maxdepth 2 -type f -name DONE    | wc -l)
  fl=$(find "$JOBS_ROOT" -mindepth 2 -maxdepth 2 -type f -name FAIL    | wc -l)
  echo "[worker] counts  READY=$rd STARTED=$st DONE=$dn FAIL=$fl"
}

claim_job() {
  # randomize order to reduce contention
  mapfile -t CANDS < <(ls -1d "$JOBS_ROOT"/* 2>/dev/null | shuf || true)
  for d in "${CANDS[@]}"; do
    [[ -d "$d" ]] || continue
    [[ -f "$d/READY" ]] || continue
    # must have an input file
    find "$d" -maxdepth 1 -type f -name '*.inp' -print -quit >/dev/null || { echo "[worker] skip: no .inp in $d" >&2; continue; }
    # atomic claim via lock directory
    if mkdir "$d/.lock" 2>/dev/null; then
      echo "[worker] claimed: $d" >&2
      mv -f "$d/READY" "$d/STARTED" 2>/dev/null || : > "$d/STARTED"
      echo "$d"           # ONLY path on stdout
      return 0
    fi
  done
  return 1
}

HEAVY_PATTERNS="${HEAVY_PATTERNS:-*.gbw *.gbw.bak *.hess *.tmp *.engrad *.cis *.cisnat *.tda *.molden *.mkl *.prop *.wfx *densities*}"
SCAN_INTERVAL="${SCAN_INTERVAL:-10}"
MAX_EMPTY_SCANS="${MAX_EMPTY_SCANS:-60}"

echo "[worker] Node=$(hostname)  OMP_NUM_THREADS=$OMP_NUM_THREADS  ORCA_BIN=$ORCA_BIN"
log_counts

empty_scans=0
while true; do
  if jobdir=$(claim_job); then
    empty_scans=0

    INPFILE="$(find "$jobdir" -maxdepth 1 -type f -name '*.inp' -print -quit)"
    if [[ -z "$INPFILE" ]]; then
      echo "[worker] ERROR: claimed $jobdir but no .inp found" >&2
      : > "$jobdir/FAIL"
      rmdir "$jobdir/.lock" 2>/dev/null || true
      continue
    fi
    BASE="${INPFILE##*/}"; BASE="${BASE%.inp}"

    # Copy entire job folder to node-local scratch (brings .xyz, etc.)
    rsync -a --delete "$jobdir"/ "$SCR"/
    cd "$SCR"

    echo "[worker] running: $ORCA_BIN ${BASE}.inp"
    set +e
    "$ORCA_BIN" "${BASE}.inp" > "${BASE}.out" 2>&1
    RC=$?
    set -e
    echo "[worker] ORCA exit=$RC"

    # Pack heavy artifacts to keep job dirs lean
    mkdir -p __packed__
    for pat in $HEAVY_PATTERNS; do
      for f in $pat; do
        [[ -f "$f" ]] && mv -f "$f" __packed__/ || true
      done
    done
    if compgen -G "__packed__/*" >/dev/null; then
      tar -czf "${BASE}.__packed__.tar.gz" __packed__ && rm -rf __packed__
    fi

    # Sync results back and mark status
    rsync -a --partial --mkpath "$SCR"/ "$jobdir"/
    if [[ $RC -eq 0 ]]; then
      : > "$jobdir/DONE"; rm -f "$jobdir/FAIL"
    else
      echo "$RC" > "$jobdir/FAIL"; rm -f "$jobdir/DONE"
    fi

    # Release lock and clean scratch for next job
    rmdir "$jobdir/.lock" 2>/dev/null || true
    find "$SCR" -mindepth 1 -maxdepth 1 -exec rm -rf {} + || true
    continue
  fi

  rm -f "$jobdir/STARTED"
  rmdir "$jobdir/.lock" 2>/dev/null || true

  # No claim this pass: report counts and decide whether to exit
  empty_scans=$((empty_scans+1))
  log_counts
  if ! find "$JOBS_ROOT" -mindepth 2 -maxdepth 2 -type f -name READY -print -quit | grep -q .; then
    echo "[worker] no READY left; exiting."
    exit 0
  fi
  if (( empty_scans >= MAX_EMPTY_SCANS )); then
    echo "[worker] waited too long without claims; exiting."
    exit 0
  fi
  sleep "$SCAN_INTERVAL"
done
