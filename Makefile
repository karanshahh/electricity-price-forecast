.PHONY: setup lint format test fetch features train backtest run_app run_api

# Default target
help:
	@echo "Available targets:"
	@echo "  setup     - Install dependencies and set up environment"
	@echo "  lint      - Run ruff linter"
	@echo "  format    - Run ruff formatter"
	@echo "  test      - Run pytest"
	@echo "  fetch     - Fetch raw data"
	@echo "  features  - Build features from raw data"
	@echo "  train     - Train model"
	@echo "  backtest  - Run backtest"
	@echo "  run_app   - Run Streamlit app"
	@echo "  run_api   - Run FastAPI server"

setup:
	pip install -e ".[dev]"
	cp -n .env.example .env 2>/dev/null || true

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

test:
	@test -f data/interim/sample_modeling.parquet || $(MAKE) sample
	PYTHONPATH=src pytest tests/ -v

fetch:
	python scripts/fetch_data.py --iso caiso

sample:
	python scripts/generate_sample_data.py

features:
	python scripts/build_features.py

train:
	python scripts/train.py

backtest:
	python scripts/backtest.py

run_app:
	streamlit run src/electricity_forecast/app/streamlit_app.py

run_api:
	uvicorn electricity_forecast.serving.api:app --reload
