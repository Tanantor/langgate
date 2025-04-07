#!/bin/sh
# Script to start Envoy with template configuration

# Path to the configuration template
TEMPLATE_PATH=${ENVOY_TEMPLATE_PATH:-/etc/envoy/envoy.yaml}
# Path to where the generated config will be stored
CONFIG_PATH=${ENVOY_CONFIG_PATH:-/tmp/envoy_config.yaml}

echo "Starting Envoy with templated configuration"
echo "Template: $TEMPLATE_PATH"
echo "Output: $CONFIG_PATH"

# Render the configuration using our Python script
if ! python3 /etc/envoy/scripts/render_config.py \
  --template "$TEMPLATE_PATH" \
  --output "$CONFIG_PATH"; then
  echo "Error: Failed to render Envoy configuration. Falling back to base config."
  exec /usr/local/bin/envoy --config-path "$TEMPLATE_PATH"
else
  echo "Successfully rendered configuration. Starting Envoy..."

  # Output the rendered configuration for debugging if DEBUG is set
  if [ "${DEBUG:-0}" = "1" ]; then
    echo "========================= RENDERED CONFIG ========================="
    # Redact sensitive info before displaying
    grep -v -i "api_key" "$CONFIG_PATH" | grep -v -i "secret"
    echo "=================================================================="
    echo "Note: Lines containing 'api_key' or 'secret' have been redacted for security."
  fi

  exec /usr/local/bin/envoy --config-path "$CONFIG_PATH"
fi
