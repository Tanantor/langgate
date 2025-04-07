#!/bin/bash
set -e

# Simple script to test Helm charts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHARTS_DIR="${SCRIPT_DIR}/../charts"

echo "Testing Helm charts (dry run)..."

# Test all charts except those in library directory
find "$CHARTS_DIR" -name Chart.yaml -not -path "$CHARTS_DIR/library/*" -print0 |
  xargs -0 -n1 dirname |
  while IFS= read -r chart; do
    echo "Testing $chart..."
    chart_name=$(basename "$chart")
    output_dir="${SCRIPT_DIR}/../rendered"
    mkdir -p "$output_dir"

    if [[ "$chart" == *"/langgate-"* && "$chart" != *"/langgate-helpers"* ]]; then
      # For component charts, use parent values but transform them for standalone use
      parent_chart="${CHARTS_DIR}/langgate"
      chart_key="${chart_name//-/_}" # Convert dash to underscore (e.g., langgate-server → langgate_server)

      echo "  Creating transformed values for standalone testing..."
      temp_values_file=$(mktemp)

      # Extract the component's values from parent chart and place at root level
      {
        echo "# Values extracted from parent chart for standalone testing"
        echo "enabled: true"

        # Add component specific values
        yq e ".${chart_key}" "$parent_chart/values.yaml"

        # Add global section
        echo "global:"
        yq e ".global" "$parent_chart/values.yaml" | sed 's/^/  /'
      } >"$temp_values_file"

      echo "  Rendering with transformed values to $output_dir/$chart_name.yaml"
      helm template test-release "$chart" --values "$temp_values_file" >"$output_dir/$chart_name.yaml"

      rm "$temp_values_file"
    else
      echo "  Rendering to $output_dir/$chart_name.yaml"
      helm template test-release "$chart" >"$output_dir/$chart_name.yaml"
    fi
  done

echo "All charts validated successfully"
