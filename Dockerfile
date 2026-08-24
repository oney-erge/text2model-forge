# The control plane intentionally stays separate from GPU-heavy ComfyUI/PyTorch.
# Compose connects it to Ollama and to an optional host ComfyUI service.
FROM python:3.12-slim-bookworm@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134

ARG TEXT2MODEL_FORGE_SOURCE_REVISION=""
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TEXT2MODEL_FORGE_SOURCE_REVISION=${TEXT2MODEL_FORGE_SOURCE_REVISION}

LABEL org.opencontainers.image.title="Text2Model Forge" \
    org.opencontainers.image.description="Auditable, human-gated orchestration for local AI-assisted 3D assets" \
    org.opencontainers.image.licenses="Apache-2.0" \
    org.opencontainers.image.version="0.2.0-rc.1" \
    org.opencontainers.image.source="https://github.com/oney-erge/text2model-forge" \
    org.opencontainers.image.revision=${TEXT2MODEL_FORGE_SOURCE_REVISION}

WORKDIR /app

# Text2Model Forge resolves adapters and other resources relative to the checkout,
# so retain the full source tree and use an editable install inside the image.
COPY . .
RUN python -m pip install --no-cache-dir --upgrade "pip==26.2.1" "setuptools==84.0.0" \
    && python -m pip install --no-cache-dir --require-hashes -r requirements.lock \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation -e . \
    && groupadd --gid 10001 text2model \
    && useradd --uid 10001 --gid text2model --no-create-home text2model \
    && mkdir -p /workspace \
    && chmod 0777 /workspace

USER text2model
EXPOSE 8766
VOLUME ["/workspace"]

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=12 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8766/doctor', timeout=2).read(1)"]

CMD ["python", "-m", "text2model_forge", "studio", "--workspace", "/workspace", "--host", "0.0.0.0", "--port", "8766", "--allow-non-loopback"]
