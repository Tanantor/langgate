from collections.abc import Generator
from typing import Any

import pytest
import pytest_asyncio
from httpx import AsyncClient as AsyncTestClient
from openai import AsyncOpenAI

from langgate.registry import ModelRegistry
from langgate.server.core.config import settings
from tests.utils.utils import random_lower_string

pytest_plugins = [
    "tests.fixtures.registry_fixtures",
    "tests.fixtures.transform_fixtures",
    "tests.fixtures.client_fixtures",
]


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances before and after each test to prevent test interference.
    This ensures tests running in parallel don't affect each other through shared state.
    """
    ModelRegistry._instance = None
    yield
    ModelRegistry._instance = None


@pytest_asyncio.fixture
def openai_client(client: AsyncTestClient) -> Generator[AsyncOpenAI, Any]:
    openai_client = AsyncOpenAI(
        api_key="",
        base_url=f"{client.base_url}",
        http_client=client,
    )
    yield openai_client


@pytest.fixture(scope="module")
def vcr_config():
    def before_record_cb(response: dict[str, Any]):
        if response["headers"].get("openai-organization"):
            response["headers"]["openai-organization"] = ["DUMMY"]
        return response

    return {
        "filter_headers": [
            ("authorization", "DUMMY"),
            ("x-api-key", "DUMMY"),
            ("openai-organization", "DUMMY"),
        ],
        "filter_query_parameters": [
            ("api_key", "DUMMY"),
            ("openai_api_key", "DUMMY"),
            ("client_secret", "DUMMY"),
        ],
        "filter_post_data_parameters": [
            ("previewToken", "DUMMY"),
            ("client_secret", "DUMMY"),
            ("refresh_token", "DUMMY"),
        ],
        "ignore_localhost": False,
        "ignore_hosts": [
            settings.TEST_SERVER_HOST,
            # tiktoken: https://community.openai.com/t/newconnectionerror-keeps-coming-up-over-a-tiktoken-file/670052/1
            "openaipublic.blob.core.windows.net",
        ],
        "record_mode": "once",  # delete cassettes or change to "all" to record new cassettes
        "before_record_response": before_record_cb,
    }


@pytest.fixture
def random_str() -> str:
    return random_lower_string()
