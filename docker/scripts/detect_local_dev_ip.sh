#!/bin/bash
# DEVELOPMENT USE ONLY
#
# This script is a local development workaround for Docker-to-host networking issues.
# It's used to detect the host machine's IP address when normal Docker networking
# methods like host.docker.internal aren't mapping correctly with Envoy DNS resolution.
# TODO: We could alternatively create a local envoy config and point to docker's internal DNS server.

case "$(uname -s)" in
Darwin)
  # macOS - try common interfaces
  for iface in en0 en1 en2; do
    IP=$(ifconfig $iface 2>/dev/null | grep "inet " | awk '{print $2}')
    if [ -n "$IP" ]; then
      echo "$IP"
      exit 0
    fi
  done
  ;;
Linux)
  # Linux - get the IP of the default route interface
  IP=$(ip -4 route get 8.8.8.8 2>/dev/null | head -1 | awk '{print $7}')
  if [ -n "$IP" ]; then
    echo "$IP"
    exit 0
  fi
  ;;
esac

echo "Error: Unable to detect local IP address.
Please set the LOCAL_DEV_IP environment variable manually."
exit 1
