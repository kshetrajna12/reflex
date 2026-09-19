# reflex server image: POST /v1/systemone on any CUDA GPU.
#
#   docker build -t reflex-server -f docker/Dockerfile .
#   docker run --gpus all -p 8000:8000 \
#     -e REFLEX_ADAPTER=<hub repo or /mounted/dir> -e REFLEX_API_KEY=<secret> \
#     -v ~/.cache/huggingface:/root/.cache/huggingface reflex-server
#
# The full python image (not -slim) is deliberate: Triton compiles the linear-attention
# kernels at first use and needs a C compiler and Python headers.
FROM python:3.12-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable && rm -rf /root/.cache/uv
ENV MODEL=Qwen/Qwen3.5-4B REFLEX_ADAPTER= REFLEX_API_KEY= SERVED_NAME=reflex PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "exec uv run --no-sync reflex-serve --model \"$MODEL\" ${REFLEX_ADAPTER:+--adapter \"$REFLEX_ADAPTER\"} --served-name \"$SERVED_NAME\" --host 0.0.0.0 --port \"$PORT\""]
