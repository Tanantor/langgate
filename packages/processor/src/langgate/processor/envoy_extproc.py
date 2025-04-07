# We would normally import these from the Envoy ext_proc SDK
# But we're creating placeholder versions instead

from collections.abc import AsyncGenerator, AsyncIterator
from logging import Logger, getLogger

from grpc import ServicerContext

logger = getLogger(__name__)


class EnvoyExtProcServicer:
    """Placeholder for EnvoyExtProcServicer."""


HttpHeaders = dict[str, str]


class HttpBody:
    """Placeholder for HttpBody."""

    def __init__(self, body: bytes = b""):
        self.body = body


class ProcessingRequest:
    """Placeholder for ProcessingRequest."""


class ProcessingResponse:
    """Placeholder for ProcessingResponse."""


class CommonResponse:
    """Placeholder for CommonResponse."""

    def __init__(self):
        self.status = "continue"
        self.headers_response = None
        self.body_response = None
        self.body_mutation = None


class BaseExtProcService(EnvoyExtProcServicer):
    """
    Base External Processor Service for Envoy.

    This is a placeholder implementation that mimics the structure of the actual
    BaseExtProcService from envoy_extproc_sdk but without direct imports.

    In the real implementation, this would inherit from the generated gRPC
    ExternalProcessorServicer class and provide implementation of the Process
    method to handle gRPC streams.
    """

    def __init__(self, name: str | None = None, logger: Logger | None = None):
        """Initialize the base ext_proc service.

        Args:
            name: Optional name for the service, used for logging
        """
        self.name = name or self.__class__.__name__
        self.logger = logger or getLogger(__name__)

    async def Process(
        self,
        request_iterator: AsyncIterator[ProcessingRequest],
        context: ServicerContext,
    ) -> AsyncGenerator[ProcessingResponse]:
        """
        Process a stream of requests from Envoy.

        This is the main entry point for the gRPC service. It would normally
        handle the stream of ProcessingRequest messages and yield ProcessingResponse
        messages back to Envoy.

        In this placeholder implementation, it's only included to show the method
        signature that would be implemented in the actual service.

        Args:
            request_iterator: An async iterator of ProcessingRequest messages
            context: The gRPC context

        Yields:
            ProcessingResponse messages for Envoy
        """
        # This implementation would normally:
        # 1. Create a request context dict to store state
        # 2. Iterate through request messages asynchronously
        # 3. Dispatch each message to the appropriate handler based on type
        # 4. Yield response messages back to Envoy

        # For this placeholder, we just log and raise an exception
        logger.error("Process_method_not_implemented")
        raise NotImplementedError("Process method not implemented in placeholder")

    # The following methods would be implemented in the real BaseExtProcService
    # They are included here as placeholders with the correct signatures

    @staticmethod
    def get_header(
        headers: HttpHeaders, name: str, lower_cased: bool = False
    ) -> str | None:
        """get a header value by name (envoy uses lower cased names)"""

    @staticmethod
    def add_header(response: CommonResponse, key: str, value: str) -> CommonResponse:
        """add a header to a CommonResponse"""
        return response

    @staticmethod
    def remove_header(response: CommonResponse, name: str) -> CommonResponse:
        """remove a header from a CommonResponse"""
        return response

    async def process_request_headers(
        self,
        headers: HttpHeaders,
        context: ServicerContext,
        request: dict,
        response: CommonResponse,
    ) -> CommonResponse:
        """
        Process request headers.
        """
        logger.debug("base_process_request_headers")
        return response

    async def process_request_body(
        self,
        body: HttpBody,
        context: ServicerContext,
        request: dict,
        response: CommonResponse,
    ) -> CommonResponse:
        """
        Process request body.
        """
        logger.debug("base_process_request_body")
        return response

    async def process_response_headers(
        self,
        headers: HttpHeaders,
        context: ServicerContext,
        request: dict,
        response: CommonResponse,
    ) -> CommonResponse:
        """
        Process response headers.
        """
        logger.debug("base_process_response_headers")
        return response

    async def process_response_body(
        self,
        body: HttpBody,
        context: ServicerContext,
        request: dict,
        response: CommonResponse,
    ) -> CommonResponse:
        """
        Process response body.
        """
        logger.debug("base_process_response_body")
        return response
