#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run the experts information generator.
# Defaults align with the Python script.

# Resolve generator relative to current working directory (icl_router root)
ROOT_DIR="$(pwd)"
PYGEN="${ROOT_DIR}/generate_experts_information.py"

# Ensure we are at icl_router root (must contain scripts/ and data/)
if [[ ! -d "${ROOT_DIR}/scripts" || ! -d "${ROOT_DIR}/data" ]]; then
  echo "Please run this script from the icl_router root directory (current: ${ROOT_DIR})" >&2
  exit 1
fi

INPUT="${ROOT_DIR}/data/train_router.json"
OUTPUT="${ROOT_DIR}/data/experts_information_500_balanced.json"
TOTAL=500
SEED=42
OVERWRITE=0

usage() {
  cat << USAGE
Usage: $(basename "$0") [options]

Options:
  -i, --input PATH        Path to train_router.json (default: ${INPUT})
  -o, --output PATH       Output JSON path (default: ${OUTPUT})
  -t, --total N           Total examples per model (default: ${TOTAL})
  -s, --seed N            Random seed (default: ${SEED})
  -f, --force             Overwrite output if exists
  -h, --help              Show this help and exit

Examples:
  $(basename "$0") -o data/experts_information_100_balanced.json -t 100 -f
  $(basename "$0") --output data/experts_information_500_balanced.json --total 500 --seed 123 --force
USAGE
}

# Parse args (supports short and long options)
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)
      INPUT="$2"; shift 2 ;;
    -o|--output)
      OUTPUT="$2"; shift 2 ;;
    -t|--total)
      TOTAL="$2"; shift 2 ;;
    -s|--seed)
      SEED="$2"; shift 2 ;;
    -f|--force|--overwrite)
      OVERWRITE=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift; break ;;
    *)
      echo "Unknown option: $1" >&2
      usage; exit 1 ;;
  esac
done

if [[ ! -f "$PYGEN" ]]; then
  echo "Generator not found: $PYGEN" >&2
  exit 1
fi

CMD=("python3" "$PYGEN" --input "$INPUT" --output "$OUTPUT" --total "$TOTAL" --seed "$SEED")
if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=("--overwrite")
fi

echo "Running: ${CMD[*]}" >&2
"${CMD[@]}"
