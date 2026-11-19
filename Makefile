COMPOSE = docker compose

.PHONY: up down logs build test lock eval shell

up:      ## build and start everything
	$(COMPOSE) up -d --build
down:
	$(COMPOSE) down
logs:
	$(COMPOSE) logs -f --tail=100 web worker
build:
	$(COMPOSE) build
test:
	$(COMPOSE) run --rm --no-deps web sh -c "cd /srv && uv sync --frozen --group dev -q && pytest -q"
lock:
	docker run --rm -v $(PWD):/w -w /w ghcr.io/astral-sh/uv:0.5.11-python3.12-bookworm-slim uv lock
eval:   ## score every fixture with the cached pipeline
	$(COMPOSE) run --rm --no-deps worker python -m app.eval.run $(ARGS)
shell:
	$(COMPOSE) exec web bash
