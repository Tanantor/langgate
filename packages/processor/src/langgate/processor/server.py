"""gRPC server for Envoy External Processing filter."""

import os

from langgate.core.logging import get_logger
from langgate.processor.ext_proc import LangGateExtProc

logger = get_logger(__name__)


def serve(*args, **kwargs):
    logger = get_logger(__name__)
    logger.info("Mock server would be running now if envoy_extproc_sdk was installed")


def run_processor_server():
    """Start the LangGate External Processor server."""
    logger.info("Starting LangGate ExtProc service...")

    port = int(os.environ.get("LANGGATE_PROC_PORT", "50051"))

    service = LangGateExtProc(name="LangGateExtProc")

    logger.info("Running server on port %s", port)
    serve(service=service, port=port)


if __name__ == "__main__":
    run_processor_server()
