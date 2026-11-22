FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
# yt-dlp needs a supported JavaScript runtime to solve YouTube challenges; an old deno is
# reported "unsupported" and YouTube then answers "Sign in to confirm you’re not a bot"
COPY --from=denoland/deno:bin-2.9.8 /deno /usr/local/bin/deno

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libharfbuzz-subset0 fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

# beat_this model weights (beats + downbeats), baked in so workers never download at run time
ADD --chmod=755 --checksum=sha256:8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331 \
    https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt /opt/models/beat_this-final0.ckpt
ENV BEAT_THIS_CHECKPOINT=/opt/models/beat_this-final0.ckpt
# BTC chord model weights (Park et al., ISMIR 2019, MIT licence), pinned to a commit and a checksum
ADD --chmod=644 --checksum=sha256:1673d23f8f9a55ae7f9e8b80a51da616debb22675b8d8b67ea6ce0ef37b0ab51 \
    https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/2682317be668032e6e4b269ded36adaa2ad57df0/test/btc_model_large_voca.pt /opt/models/btc_model_large_voca.pt
ENV BTC_CHECKPOINT=/opt/models/btc_model_large_voca.pt
# Whisper "tiny" (OpenAI, MIT licence): hears where each word of the lyrics is sung
ADD --chmod=644 --checksum=sha256:65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9 \
    https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt /opt/models/whisper-tiny.pt
ENV WHISPER_MODEL=/opt/models/whisper-tiny.pt

WORKDIR /srv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY app ./app
COPY tests ./tests
COPY alembic.ini ./
COPY migrations ./migrations

RUN useradd --uid 1000 --create-home unplug
USER unplug
