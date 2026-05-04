#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train_resume.sh — Auto-resume training until step 10000 checkpoint exists.
#
# Usage:
#   chmod +x scripts/train_resume.sh
#   ./scripts/train_resume.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

CHECKPOINT_DIR="checkpoints/pi05_ur10_lora/pi05_ur10_v2"
DONE_STEP="10000"
DONE_MARKER="${CHECKPOINT_DIR}/${DONE_STEP}"

TRAIN_CMD=(
    env
    CUDA_VISIBLE_DEVICES=0,1
    LEROBOT_VIDEO_BACKEND=pyav
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.86
    uv run scripts/train.py pi05_ur10_lora
    --exp-name pi05_ur10_v2
    --no-wandb-enabled
    --batch-size 96
    --resume
)

ATTEMPT=0

while true; do
    # ── Done check ───────────────────────────────────────────────────────────
    if [ -d "${DONE_MARKER}" ]; then
        echo ""
        echo "✅  Step ${DONE_STEP} checkpoint found at ${DONE_MARKER}"
        echo "    Training complete — exiting."
        exit 0
    fi

    ATTEMPT=$(( ATTEMPT + 1 ))
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Show latest saved checkpoint so we know where we resume from
    LATEST=$(ls -1d "${CHECKPOINT_DIR}"/[0-9]* 2>/dev/null | sort -V | tail -1 || echo "none")
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Attempt #${ATTEMPT}  |  ${TIMESTAMP}"
    echo "  Latest checkpoint : ${LATEST}"
    echo "  Target            : ${DONE_MARKER}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── Launch training ──────────────────────────────────────────────────────
    set +e
    "${TRAIN_CMD[@]}"
    EXIT_CODE=$?
    set -e

    # ── Exit-code handling ───────────────────────────────────────────────────
    if [ ${EXIT_CODE} -eq 0 ]; then
        if [ -d "${DONE_MARKER}" ]; then
            echo "✅  Training finished cleanly and step ${DONE_STEP} checkpoint exists."
            exit 0
        else
            echo "⚠️  Training exited cleanly but step ${DONE_STEP} not found."
            echo "    The run may have ended early. Check your config's num_train_steps."
            exit 0
        fi
    fi

    # SIGKILL / OOM shows as exit 137 (128 + 9)
    if [ ${EXIT_CODE} -eq 137 ]; then
        echo "💀  OOM-killed (exit 137). Waiting 15 s for GPU memory to free..."
        sleep 15
    else
        echo "💥  Crashed (exit ${EXIT_CODE}). Waiting 10 s before retry..."
        sleep 10
    fi
done