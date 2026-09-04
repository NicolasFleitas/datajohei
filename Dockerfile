# Runtime image for the DataJohei Streamlit app.
# Uses the official uv base image so `uv sync --frozen` reproduces uv.lock exactly.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Dependencies first: this layer is cached until pyproject.toml/uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Application source (the project itself runs from source, not as an installed package).
COPY src/ ./src/
COPY app.py main.py ./
COPY .streamlit/ ./.streamlit/

ENV PATH="/app/.venv/bin:$PATH" \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

RUN useradd --create-home --uid 1001 appuser
USER appuser

EXPOSE 8501

# Streamlit's built-in health endpoint; port read from env so runtime overrides keep working.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, sys, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:' + os.environ['STREAMLIT_SERVER_PORT'] + '/_stcore/health', timeout=4); sys.exit(0 if r.status == 200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
