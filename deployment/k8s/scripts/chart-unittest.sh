#!/bin/bash
set -e

# Simple script to run unit tests for Helm charts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHARTS_DIR="${SCRIPT_DIR}/../charts"

echo "Running unit tests for Helm charts..."

# Test all charts with tests directory
find "$CHARTS_DIR" -name Chart.yaml -print0 |
  xargs -0 -n1 dirname |
  while IFS= read -r chart; do
    if [ -d "$chart/tests" ]; then
      echo "Testing $chart..."
      helm unittest -3 "$chart"
    else
      echo "No tests found for $chart, skipping..."
    fi
  done

echo "All unit tests completed"
