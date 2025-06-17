"""External Processor implementation for Envoy."""

import json

from grpc import ServicerContext

from langgate.core.logging import get_logger
from langgate.processor.envoy_extproc import (
    BaseExtProcService,
    CommonResponse,
    HttpBody,
    HttpHeaders,
)
from langgate.transform import LocalTransformerClient

logger = get_logger(__name__)

PROXY_HEADER = "x-langgate"
TARGET_ENDPOINT = "/v1"
ROUTE_HEADER = "x-route-to"


class LangGateExtProc(BaseExtProcService):
    """
    External Processor for Envoy.
    Implements the Envoy ext_proc External Processing Filter proto.

    This class processes HTTP requests and responses by implementing
    handlers for different parts of the request/response cycle.

    Note: This is a placeholder implementation that mimics the structure
    of a real ext_proc implementation but without direct imports from
    the envoy_extproc_sdk which needs to be updated.

    The envoy_extproc_sdk is based on the gRPC generated code from
    Envoy's External Processing filter proto definitions, defined in:
    https://github.com/envoyproxy/envoy/blob/main/api/envoy/service/ext_proc/v3/external_processor.proto

    TODO: Upgrade Envoy in https://github.com/Tanantor/envoy-extproc-sdk
    to address CVEs before implementing this service.
    This will involve significant refactoring of envoy-extproc-sdk to
    implement an architecture like the awesome Go ext_proc SDK by the same author.

    We may opt to use Go for this implementation instead of Python.
    Alternatively we may even use PyO3 for Rust-Python bindings.

    Tools like https://github.com/go-python/gopy for generating Python extension modules from Go
    could be used to create a Python wrapper around the Go tranformers for the local transformer setup.
    Another approach is building Go code as shared libraries using cgo and loading them in Python via ctypes.
    This would require packaging pre-compiled binaries for different platforms to maintain the simplicity
    of pip installation for end users.
    """

    def __init__(self, name: str = "LangGateExtProc"):
        """Initialize the processor."""
        super().__init__(name=name)
        self.transformer = LocalTransformerClient()
        logger.info("langgate_ext_proc_service_initialized")

    async def process_request_headers(
        self,
        headers: HttpHeaders,
        context: ServicerContext,
        request: dict,
        response: CommonResponse,
    ) -> CommonResponse:
        # Skip processing if not specifically targeting v1 endpoint
        if not request.get("path", "").startswith(TARGET_ENDPOINT):
            return response

        # Store content-type to check if JSON in body phase
        content_type = self.get_header(headers, "content-type")
        request["content_type"] = content_type

        # Store the original path for possible rewriting later
        request["original_path"] = request.get("path", "")
        self.logger.debug(f"Setting original path context: {request['original_path']}")

        # Replace the Authorization header
        # This is a placeholder for the actual implementation
        # TODO: We need to implement how we are actually doing this with the transformer
        auth_header = self.get_header(headers, "authorization")
        if auth_header:
            self.remove_header(response, "authorization")
            self.add_header(response, "authorization", "Bearer sk-proxy-key-replaced")
            self.logger.debug("Replaced Authorization header")

        return response

    async def process_request_body(
        self,
        body: HttpBody,
        context: ServicerContext,
        request: dict,
        response: CommonResponse,
    ) -> CommonResponse:
        original_path = request.get("original_path", "")
        if not original_path.startswith(TARGET_ENDPOINT):
            return response

        if (
            request.get("content_type")
            and "application/json" in request["content_type"]
        ):
            try:
                body_str = body.body.decode("utf-8")
                self.logger.debug(f"ORIGINAL BODY: {body_str}")
                json_body = json.loads(body_str)

                # Track modifications
                modified = False

                # Extract and process the 'model' parameter if present
                if "model" in json_body:
                    original_model = json_body["model"]
                    request["original_model"] = original_model

                    # Transform parameters using our transformer
                    _, transformed_params = await self.transformer.get_params(
                        original_model, json_body
                    )

                    # Replace the original JSON with transformed parameters
                    modified = True
                    self.logger.debug(
                        f"Transforming parameters for model: {original_model}"
                    )

                    target_route = transformed_params.get("base_url", "")
                    self.add_header(response, ROUTE_HEADER, target_route)

                    # We can't directly set host/authority headers,
                    # but Envoy will use x-route-to for host rewriting
                    # based on our route configuration's host_rewrite_header

                    # Save target route for path rewriting
                    request["target_route"] = target_route

                    # Clear the route cache for the current client request.
                    # We modified headers that are used to calculate the route.
                    response.clear_route_cache = True  # type: ignore

                if modified:
                    # TODO: transformed params will contain some headers, not all of it is to be sent as body
                    # the current transformer is returning params for a LangChain BaseChatModel.
                    new_body = json.dumps(transformed_params).encode("utf-8")

                    self.logger.debug(f"MODIFIED BODY SENT: {new_body.decode('utf-8')}")

                    response.body_mutation.body = new_body  # type: ignore

                    # When working with Envoy ExtProc, we don't set the content-length header
                    # as Envoy uses transfer-encoding: chunked when body is modified.
                    self.add_header(response, "x-content-length", str(len(new_body)))
                    request["body_modified"] = True

            except (json.JSONDecodeError, UnicodeDecodeError):
                self.logger.warning("Failed to decode JSON body")
        return response

    async def process_response_headers(
        self,
        headers: HttpHeaders,
        context: ServicerContext,
        request: dict,
        response: CommonResponse,
    ) -> CommonResponse:
        original_path = request.get("original_path", "")
        if not original_path.startswith(TARGET_ENDPOINT):
            return response

        self.logger.debug(
            f"""Processing response headers for path: {original_path}
            with context: {request} and headers: {headers}"""
        )
        # Add a marker header to indicate this request was proxied
        self.add_header(response, PROXY_HEADER, "true")

        return response
