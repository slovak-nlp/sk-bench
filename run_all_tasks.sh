#!/bin/bash

# Run all task files sequentially from a specified folder
# Stops if any task fails

# export API_KEY=""
export WANDB_PROJECT=skbench
export WANDB_MODE=offline

MODEL_NAME="openrouter/google/gemini-3.1-flash-lite-preview"
BASE_URL="https://openrouter.ai/api/v1"
TASKS_FOLDER="./thinking_tasks_zero_by_lang"

# Files to exclude (comma-separated, filenames only without path)
EXCLUDE_LIST="belebele_sk.txt,arc_sk.txt,arc_en.txt"

# Get all .txt files in the folder
TASK_FILES=$(find "$TASKS_FOLDER" -name "*.txt" | sort)

if [ -z "$TASK_FILES" ]; then
    echo "No .txt files found in $TASKS_FOLDER"
    exit 1
fi

# Convert exclude list to array
IFS=',' read -ra EXCLUDE_ARRAY <<< "$EXCLUDE_LIST"

# Filter out excluded files
FILTERED_FILES=""
TOTAL_FILES=0
SKIPPED_FILES=0

for FILE in $TASK_FILES; do
    FILENAME=$(basename "$FILE")
    SKIP=false

    for EXCLUDE in "${EXCLUDE_ARRAY[@]}"; do
        if [ "$FILENAME" == "$EXCLUDE" ]; then
            SKIP=true
            ((SKIPPED_FILES++))
            echo "[$(date '+%H:%M:%S')] EXCLUDED: $FILENAME"
            break
        fi
    done

    if [ "$SKIP" = false ]; then
        FILTERED_FILES="$FILTERED_FILES $FILE"
        ((TOTAL_FILES++))
    fi
done

echo "Found $TOTAL_FILES task files to process ($SKIPPED_FILES excluded)"
echo "========================================"

for TASK_FILE in $FILTERED_FILES; do
    FILENAME=$(basename "$TASK_FILE")
    echo ""
    echo "[$(date '+%H:%M:%S')] Running: $FILENAME"
    echo "----------------------------------------"

    uv run python -m eval_litellm \
        --model-name "$MODEL_NAME" \
        --base-url "$BASE_URL" \
        --tasks-file "$TASK_FILE" \
        --save-details \
        --wandb \
        --max-tokens 8192 \
        --reasoning-effort minimal

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "[$(date '+%H:%M:%S')] FAILED: $FILENAME (exit code: $EXIT_CODE)"
        echo "Stopping execution due to failure."
        exit 1
    fi

    echo "[$(date '+%H:%M:%S')] SUCCESS: $FILENAME"
done

echo ""
echo "========================================"
echo "All tasks completed successfully!"
exit 0
