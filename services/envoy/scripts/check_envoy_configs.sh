#!/bin/bash
# Script to validate Envoy configuration with Jinja2 templates
# Usage: ./check_envoy_configs.sh config_template.yaml [var1=value1 var2=value2 ...]

set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 config_template.yaml [var1=value1 var2=value2 ...]"
  echo "Validates an Envoy configuration with template variables"
  exit 1
fi

CONFIG_TEMPLATE=$1
shift

# Set environment variables for our template renderer
for var in "$@"; do
  key=$(echo "$var" | cut -d= -f1)
  value=$(echo "$var" | cut -d= -f2)
  export "$key=$value"
done

# Path to the rendered config
RENDERED_CONFIG="/tmp/envoy_check_config.yaml"

# Use our Python script to render the config
echo "Rendering template $CONFIG_TEMPLATE..."
if ! python3 "$(dirname "$0")/render_config.py" \
  --template "$CONFIG_TEMPLATE" \
  --output "$RENDERED_CONFIG"; then
  echo "Error: Failed to render Envoy configuration." >&2
  exit 1
fi

# Check if envoy is installed
if ! command -v envoy &>/dev/null; then
  echo "Warning: Envoy binary not found. To validate using Docker, run:" >&2
  echo "docker run --rm -v \$PWD:/config envoyproxy/envoy:v1.33-latest -c /config/$(basename "$RENDERED_CONFIG") --mode validate" >&2

  # Create Docker command for validation
  docker_cmd="docker run --rm -v $(dirname "$RENDERED_CONFIG"):/config envoyproxy/envoy:v1.33-latest -c /config/$(basename "$RENDERED_CONFIG") --mode validate"

  echo "Running Docker validation: $docker_cmd" >&2
  if eval "$docker_cmd"; then
    echo "Configuration validation successful." >&2
    exit 0
  else
    echo "Error: Configuration is invalid." >&2
    exit 1
  fi
else
  echo "Validating configuration..." >&2

  # Validate with local Envoy
  if envoy -c "$RENDERED_CONFIG" --mode validate; then
    echo "Configuration validation successful." >&2
    exit 0
  else
    echo "Error: Configuration is invalid." >&2
    exit 1
  fi
fi
