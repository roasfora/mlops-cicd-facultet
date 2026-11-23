# Linux/macOS commands. Windows equivalents are in README.md.
PYTHON ?= python

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) src/prepare_data.py

train:
	$(PYTHON) src/train.py

test:
	$(PYTHON) -m pytest -v

# What CI runs: train first (tests need model.joblib), then all tests
ci: train test

serve:
	$(PYTHON) -m uvicorn src.app:app --reload

.PHONY: install data train test ci serve
