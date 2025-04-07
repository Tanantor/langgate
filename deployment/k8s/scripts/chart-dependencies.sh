#!/bin/bash
set -e

# Script to update chart dependencies

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHARTS_DIR="${SCRIPT_DIR}/../charts"

echo "Updating chart dependencies..."

# Process all charts (not in library) that have dependencies
find "$CHARTS_DIR" -name Chart.yaml -not -path "$CHARTS_DIR/library/*" -print0 |
  xargs -0 -n1 dirname |
  while IFS= read -r chart; do
    if grep -q "dependencies:" "$chart/Chart.yaml"; then
      echo "Updating dependencies for $chart..."
      helm dependency update "$chart"
    fi
  done

echo "All dependencies updated"
