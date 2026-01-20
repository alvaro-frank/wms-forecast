PYTHON := py -3.10
VENV_BIN := .venv\Scripts
PY  := $(VENV_BIN)\python.exe
MKVENV := if not exist .venv ( $(PYTHON) -m venv .venv )
RMVENV := if exist .venv rmdir /s /q .venv
MAIN_SCRIPT = src/main.py
PORT = 5000
RESUME ?= False

RESUME_FLAG = 
ifeq ($(RESUME),True)
	RESUME_FLAG = --resume
endif

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
	$(PY) -m dvc pull

pull-data:
	$(PY) -m dvc pull

clean:
	$(RMVENV)

train:
	$(PY) $(MAIN_SCRIPT) train --model $(MODEL) $(RESUME_FLAG) $(ARGS)

evaluate:
	$(PY) $(MAIN_SCRIPT) evaluate --model $(MODEL) $(if $(BRAND),--brand $(BRAND),)

test:
	$(PY) $(MAIN_SCRIPT) test --model $(MODEL) --hierarchy $(HIER) --brand $(BRAND)

mlflow:
	$(VENV_BIN)\mlflow ui --port $(PORT)

unit-test:
	$(PY) -m pytest src/test/

all: clean setup unit-test train evaluate test

.DEFAULT_GOAL := help