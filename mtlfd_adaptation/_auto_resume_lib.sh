# Sourced by train_mtlfd_*.sh launchers to survive the gpuq partition's 7h wall-clock limit
# without needing a human (or a live Claude Code session watching squeue) to notice a TIMEOUT and
# manually resubmit with --resume. See docs/03_test_adaptation.md-style rationale: earlier runs
# needed 4-5 manual resubmissions over several days; a background "watcher" tied to a chat session
# was tried next but dies whenever that session restarts (observed twice in one day) - this is
# SLURM-native instead, so it doesn't depend on anything outside the job itself.
#
# Mechanism: the launcher adds `#SBATCH --signal=B:USR1@300`, which makes SLURM send SIGUSR1 to
# this script 300s (5 min) before the job's time limit - enough warning to resubmit before the
# hard kill. The launcher must, after sourcing this file:
#   1. set OUTPUT_LOG to the exact path its own `#SBATCH --output=` resolves to (with $SLURM_JOB_ID
#      substituted for %j) - used to recover this run's checkpoint dir from the "save path: ..."
#      line train_mtlfd.py prints at startup.
#   2. trap _auto_resume_on_timeout USR1
#   3. launch the training python process in the BACKGROUND (`... &`) and `wait "$!"` on it, not
#      run it in the foreground - bash only runs trap handlers between commands / while `wait` is
#      blocked, not while a foreground child has control of the terminal.
#
# STEP_CAP: once this run's checkpoint dir's latest model_save-<N>.pt reaches this many steps, the
# trap stops resubmitting (further training past that point becomes a deliberate decision, not
# indefinite auto-continuation) - just lets the job die at the time limit as normal.
STEP_CAP="${STEP_CAP:-400000}"

_auto_resume_max_step() {
    ls "$1" 2>/dev/null | grep -oE 'model_save-[0-9]+\.pt' | grep -oE '[0-9]+' | sort -n | tail -1
}

_auto_resume_on_timeout() {
    # the caller runs under `set -e`; don't let any single failed check inside this handler (e.g.
    # grep matching nothing) abort the trap before it reaches the actual resubmit call below.
    set +e
    echo "[auto-resume] caught SIGUSR1 (~5 min before time limit) at $(date -Iseconds)"

    local ckpt_dir
    ckpt_dir=$(grep -oE '^save path: {2}.*' "$OUTPUT_LOG" 2>/dev/null | tail -1 | sed 's/^save path: *//')
    if [ -z "$ckpt_dir" ] || [ ! -d "$ckpt_dir" ]; then
        echo "[auto-resume] could not find a 'save path:' line in $OUTPUT_LOG - NOT resubmitting" \
             "(this run may not have gotten far enough to start training; check manually)"
        return
    fi

    local step
    step=$(_auto_resume_max_step "$ckpt_dir")
    if [ -n "$step" ] && [ "$step" -ge "$STEP_CAP" ]; then
        echo "[auto-resume] $ckpt_dir already at step $step >= STEP_CAP=$STEP_CAP - not resubmitting"
        return
    fi

    echo "[auto-resume] resubmitting $ckpt_dir (last checkpoint step: ${step:-none}) with --resume"
    sbatch_out=$(EXPERIMENT_FILE="$ckpt_dir" sbatch "$LAUNCHER_SCRIPT" --resume 2>&1)
    echo "[auto-resume] sbatch output: $sbatch_out"
}
