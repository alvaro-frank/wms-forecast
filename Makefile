PYTHON := py -3.10
VENV_BIN := .venv\Scripts
PY  := $(VENV_BIN)\python.exe
MKVENV := if not exist .venv ( $(PYTHON) -m venv .venv )
RMVENV := if exist .venv rmdir /s /q .venv
MAIN_SCRIPT = src/main.py

help:
	@echo "WMS Forecast - Available commands:"
	@echo "  make setup      - Create and activate virtual environment"
	@echo "  make train      - Train the model"
	@echo "  make evaluate   - Evaluate the model"
	@echo "  make venv       - Create and activate virtual environment"
	@echo "  make clean      - Remove virtual environment and cache files"

setup:
	$(MKVENV)
	$(PY) -m pip install -r requirements.txt

clean:
	if exist .venv rmdir /s /q .venv

train:
	python $(MAIN_SCRIPT) train $(ARGS)

evaluate:
	python $(MAIN_SCRIPT) evaluate --model $(MODEL) $(if $(BRAND),--brand $(BRAND),)

test:
	python $(MAIN_SCRIPT) test --model $(MODEL) --hierarchy $(HIER) --brand $(BRAND)

all: clean setup train evaluate test

.DEFAULT_GOAL := help