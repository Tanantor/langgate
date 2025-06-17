#!/bin/bash
set -e

# Script to test Helm charts with sample configuration

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHARTS_DIR="${SCRIPT_DIR}/../charts"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Config file path
LANGGATE_CONFIG="${REPO_ROOT}/examples/langgate_config.yaml"
OUTPUT_DIR="${SCRIPT_DIR}/../rendered"
mkdir -p "$OUTPUT_DIR"

# Sample model config for testing
SAMPLE_MODELS='{
  "anthropic/claude-sonnet-4-0": {
    "name": "Claude-4 Sonnet",
    "service_provider": "anthropic"
  }
}'

echo "Testing Helm charts with sample config (dry run)..."
echo "  Config path: $LANGGATE_CONFIG"
echo "  Using sample models configuration"
echo "  Writing to $OUTPUT_DIR/langgate-with-config.yaml"

# Create temp file for models
temp_models_file=$(mktemp)
echo "$SAMPLE_MODELS" >"$temp_models_file"

# Render template with sample configuration
if [ -f "$LANGGATE_CONFIG" ]; then
  helm template test-release "${CHARTS_DIR}/langgate" \
    --set-file config.data.langgate_config="$LANGGATE_CONFIG" \
    --set-file config.data.langgate_models="$temp_models_file" \
    >"$OUTPUT_DIR/langgate-with-config.yaml"
else
  echo "Warning: Config file not found at $LANGGATE_CONFIG"
  helm template test-release "${CHARTS_DIR}/langgate" >"$OUTPUT_DIR/langgate-with-config.yaml"
fi

# Clean up
rm "$temp_models_file"

echo "Chart testing with config completed"
