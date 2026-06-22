.PHONY: install playground run-service

# Prepend the virtual environment's Scripts directory to PATH for Windows support
ifeq ($(OS),Windows_NT)
export PATH := $(CURDIR)/.venv/Scripts;$(CURDIR)/.venv/Scripts:$(PATH)
endif

install:
	uv pip install -e .

playground:
	uvicorn expense_agent.web_service:app --host 127.0.0.1 --port 8080

run-service:
	uvicorn expense_agent.web_service:app --host 127.0.0.1 --port 8080

generate-traces:
	uv run python tests/eval/generate_traces.py

grade:
	uv run python tests/eval/grade_traces.py



