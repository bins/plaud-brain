.PHONY: dev db-up db-down install test check sync run-once whisper-build whisper-model

install:
	python3.12 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -e '.[dev]'

db-up:
	docker compose up -d postgres

db-down:
	docker compose down

dev: db-up

check:
	./scripts/check_host.sh

test:
	.venv/bin/pytest -q

sync:
	.venv/bin/plaud-brain discover

run-once:
	.venv/bin/plaud-brain run-once

whisper-build:
	./scripts/build_whisper_vulkan.sh

whisper-model:
	./scripts/download_whisper_model.sh
