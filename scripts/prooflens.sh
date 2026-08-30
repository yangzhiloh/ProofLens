#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

command_name="${1:-help}"
python_version="${2:-3.11}"
output="${3:-artifacts/demo}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install uv 0.12.0, then rerun this command." >&2
    exit 1
fi

setup() {
    uv sync --locked --extra dev --python "$python_version"
}

publish_artifacts() {
    setup
    uv run --locked --extra dev python scripts/reproduce_small.py \
        --output "$output" --experiment e4 --publish-demo-artifacts
}

case "$command_name" in
    setup)
        setup
        ;;
    verify)
        setup
        uv run --locked --extra dev python -m ruff check src tests scripts
        uv run --locked --extra dev python -m pytest -q
        uv run --locked --extra dev python scripts/release_check.py --root .
        ;;
    preflight)
        setup
        uv run --locked --extra dev python scripts/task8_preflight.py
        ;;
    artifacts)
        publish_artifacts
        ;;
    demo)
        if [[ ! -f "$output/export/artifact_manifest.json" ]]; then
            publish_artifacts
        fi
        uv run --locked --extra dev python -m prooflens.cli app \
            --backend onnx \
            --model "$output/export/model.onnx" \
            --calibration "$output/export/calibration.json"
        ;;
    help|-h|--help)
        printf '%s\n' \
            "ProofLens one-click workflow" \
            "  bash scripts/prooflens.sh setup     # install the locked environment" \
            "  bash scripts/prooflens.sh verify    # lint, test, and run the release gate" \
            "  bash scripts/prooflens.sh preflight # audit task 8 readiness without downloads" \
            "  bash scripts/prooflens.sh artifacts # generate the fixture demo bundle" \
            "  bash scripts/prooflens.sh demo      # generate if needed, then launch the app"
        ;;
    *)
        echo "unknown command: $command_name" >&2
        exit 2
        ;;
esac
