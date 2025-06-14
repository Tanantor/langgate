from decimal import Decimal
from typing import Annotated

from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    PlainSerializer,
    PlainValidator,
    TypeAdapter,
)

AnyHttpUrlAdapter = TypeAdapter(AnyHttpUrl)

# https://github.com/pydantic/pydantic/issues/7186
HttpUrlStr = Annotated[
    str,
    PlainValidator(lambda x: AnyHttpUrlAdapter.validate_strings(x)),
    AfterValidator(lambda x: str(x).rstrip("/")),
]


def validate_url_or_env_var(v: str) -> str:
    """Validate a string as either a URL or an environment variable reference."""
    if v is None:
        return v
    if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
        return v
    return str(AnyHttpUrlAdapter.validate_strings(v)).rstrip("/")


# Type that can be either a URL or an environment variable reference
UrlOrEnvVar = Annotated[str, AfterValidator(validate_url_or_env_var)]


NormalizedDecimal = Annotated[
    Decimal,
    PlainSerializer(lambda v: str(v.normalize()), return_type=str, when_used="json"),
]
