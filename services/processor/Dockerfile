ARG USER="langgate"
ARG WORKDIR="/home/${USER}"
FROM --platform=$TARGETPLATFORM python:3.13-slim-bookworm AS base

FROM base AS build
ARG USER
ARG WORKDIR
WORKDIR ${WORKDIR}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONPATH=${WORKDIR} \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/opt/.venv/bin/python3.13 \
    UV_PROJECT_ENVIRONMENT=/opt/.venv

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN groupadd -r ${USER} && useradd -r -g ${USER} --shell /bin/bash ${USER} \
    && apt-get update && apt-get install -y --no-install-recommends \
        curl=7.88.* \
        ca-certificates=20230311 \
        build-essential=12.9* \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache/* /usr/share/doc/*

COPY --from=ghcr.io/astral-sh/uv:0.6.8 /uv /usr/local/bin/uv

RUN python -m venv /opt/.venv
ENV PATH="/opt/.venv/bin:$PATH"

# Install external dependencies
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/core/pyproject.toml,target=packages/core/pyproject.toml \
    --mount=type=bind,source=packages/registry/pyproject.toml,target=packages/registry/pyproject.toml \
    --mount=type=bind,source=packages/server/pyproject.toml,target=packages/server/pyproject.toml \
    --mount=type=bind,source=packages/client/pyproject.toml,target=packages/client/pyproject.toml \
    --mount=type=bind,source=packages/transform/pyproject.toml,target=packages/transform/pyproject.toml \
    --mount=type=bind,source=packages/processor/pyproject.toml,target=packages/processor/pyproject.toml \
    --mount=type=bind,source=packages/sdk/pyproject.toml,target=packages/sdk/pyproject.toml \
    uv pip compile pyproject.toml \
        --extra processor \
        --no-emit-package langgate-core \
        --no-emit-package langgate-registry \
        --no-emit-package langgate-server \
        --no-emit-package langgate-client \
        --no-emit-package langgate-registry \
        --no-emit-package langgate-transform \
        --no-emit-package langgate-processor \
        --no-emit-package langgate-sdk \
        -o third-party-requirements.txt && \
    uv pip install -r third-party-requirements.txt

# Copy the processor package and workspace dependencies
COPY services/processor/scripts scripts/
COPY packages/core/ packages/core/
COPY packages/transform/ packages/transform/
COPY packages/processor/ packages/processor/

# Sync the package and workspace dependencies
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/core/pyproject.toml,target=packages/core/pyproject.toml \
    --mount=type=bind,source=packages/registry/pyproject.toml,target=packages/registry/pyproject.toml \
    --mount=type=bind,source=packages/server/pyproject.toml,target=packages/server/pyproject.toml \
    --mount=type=bind,source=packages/client/pyproject.toml,target=packages/client/pyproject.toml \
    --mount=type=bind,source=packages/transform/pyproject.toml,target=packages/transform/pyproject.toml \
    --mount=type=bind,source=packages/processor/pyproject.toml,target=packages/processor/pyproject.toml \
    --mount=type=bind,source=packages/sdk/pyproject.toml,target=packages/sdk/pyproject.toml \
    uv sync \
        --locked \
        --no-dev \
        --no-editable \
        --package langgate-core && \
    uv sync \
        --locked \
        --no-dev \
        --no-editable \
        --package langgate-transform && \
    uv sync \
        --locked \
        --no-dev \
        --no-editable \
        --package langgate-processor

RUN chown -R ${USER}:${USER} ./

FROM base AS runner
ARG USER
ARG WORKDIR
WORKDIR ${WORKDIR}

ARG MAINTAINER="saran@tanantor.com"
LABEL maintainer=${MAINTAINER}

ENV WORKDIR=${WORKDIR} \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONPATH=${WORKDIR} \
    PATH="/opt/.venv/bin:$PATH"

RUN groupadd -r ${USER} && useradd -r -g ${USER} ${USER} \
    && apt-get update && apt-get install -y --no-install-recommends \
        tini=0.19* \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache/* /usr/share/doc/*

USER ${USER}
EXPOSE 50051

COPY --from=build /opt/.venv /opt/.venv
COPY --from=build ${WORKDIR} .

ENTRYPOINT ["tini", "-g", "--"]
CMD ["./scripts/start.sh"]
