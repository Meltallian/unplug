FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
# yt-dlp needs a JavaScript runtime for YouTube extraction
COPY --from=denoland/deno:bin-2.1.4 /deno /usr/local/bin/deno

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libharfbuzz-subset0 fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

# beat_this model weights (beats + downbeats), baked in so workers never download at run time
ADD --checksum=sha256:8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331 \
    https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt /opt/models/beat_this-final0.ckpt
RUN chmod 644 /opt/models/beat_this-final0.ckpt
ENV BEAT_THIS_CHECKPOINT=/opt/models/beat_this-final0.ckpt

WORKDIR /srv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY app ./app
COPY tests ./tests
COPY alembic.ini ./
COPY migrations ./migrations

RUN useradd --uid 1000 --create-home unplug
USER unplug
