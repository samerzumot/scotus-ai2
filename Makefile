.PHONY: venv install hotrun run

VENV_DIR := .venv
PY := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

venv:
	python3 -m venv $(VENV_DIR)

install: venv
	$(PIP) install -r requirements.txt

hotrun: install
	$(PY) hotrun.py

run: install
	$(PY) -m hypercorn app:app --bind 0.0.0.0:8000


