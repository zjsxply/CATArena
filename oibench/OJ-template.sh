#!/bin/bash
# Usage: bash OJ.sh <problem_id> <program_path> [--format json]
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYBIN=$(command -v python3 || command -v python)
PROBLEM_ID=$1
PROGRAM_PATH=$2
if [ -z "$PROBLEM_ID" ] || [ -z "$PROGRAM_PATH" ]; then
  echo "Usage: bash OJ.sh <problem_id> <program_path> [--format json]" >&2
  exit 1
fi
shift 2
META_PATH="__OJ_META_PATH__"
RUNNER="__OJ_RUNNER_PATH__"
"$PYBIN" "$RUNNER" --problem-id "$PROBLEM_ID" --program-path "$PROGRAM_PATH" --workdir "$PWD" --meta-path "$META_PATH" "$@"
