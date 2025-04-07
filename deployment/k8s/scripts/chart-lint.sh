#!/bin/bash
set -e

# Simple script to lint Helm charts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHARTS_DIR="${SCRIPT_DIR}/../charts"

echo "Linting Helm charts..."

# Lint all charts except those in library directory
find "$CHARTS_DIR" -name Chart.yaml -not -path "$CHARTS_DIR/library/*" -print0 |
  xargs -0 -n1 dirname |
  while IFS= read -r chart; do
    echo "Linting $chart..."
    if [[ "$chart" == *"/langgate-"* && "$chart" != *"/langgate-helpers"* ]]; then
      # For component charts, use parent values
      parent_chart="${CHARTS_DIR}/langgate"
      helm lint "$chart" --values "$parent_chart/values.yaml"
    else
      helm lint "$chart"
    fi
  done

echo "All charts linted successfully"
