#!/bin/bash
set -e

export APP_MODULE=${APP_MODULE:-"langgate.server.main:app"}

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-4000}
LOG_LEVEL=${LOG_LEVEL:-info}

if [[ $APP_ENV == "local"* ]]; then
    exec uvicorn "$APP_MODULE" --reload --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
else
    exec uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
fi
