#!/usr/bin/env bash
set -euo pipefail

DATA="all"
MIN_MATCH_RATE="0.98"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA="$2"; shift 2 ;;
    --min-match-rate) MIN_MATCH_RATE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"
python features/build_train_only_features.py --data "$DATA" --min-match-rate "$MIN_MATCH_RATE"
