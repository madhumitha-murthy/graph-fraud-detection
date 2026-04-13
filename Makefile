PYTHON := venv/bin/python
JAVA_OPTS := -Djava.security.manager=allow

.PHONY: install features train-gae train-graphsage train api test

install:
	python -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
	venv/bin/pip install torch-geometric -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
	venv/bin/pip install -r requirements.txt

data:
	$(PYTHON) data/download_data.py

features:
	JAVA_TOOL_OPTIONS="$(JAVA_OPTS)" $(PYTHON) -m src.features.spark_feature_engineering

train-gae:
	$(PYTHON) -m src.training.train_gae

train-graphsage:
	$(PYTHON) -m src.training.train_graphsage

train: train-gae train-graphsage

api:
	venv/bin/uvicorn src.api.main:app --reload --port 8000

test:
	venv/bin/pytest tests/ -v --cov=src --cov-report=term-missing
