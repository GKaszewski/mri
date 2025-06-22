FROM node:latest AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:bookworm-slim AS backend
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV APP_NAME=mri
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY web.py pyproject.toml uv.lock .python-version best_brain_tumor_resnet18.pth ./
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

EXPOSE 8000
ENTRYPOINT [ "uv", "run", "uvicorn", "web:app", "--host", "0.0.0.0"]