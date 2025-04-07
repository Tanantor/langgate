#!/bin/bash
set -e

export LANGGATE_PROC_PORT=${LANGGATE_PROC_PORT:-50051}
export LOG_LEVEL=${LOG_LEVEL:-info}

exec python -m langgate.processor.server
