from pathlib import Path

FILES = """
README.md LICENSE .gitignore .env.example .pre-commit-config.yaml pyproject.toml Makefile
configs/base.yaml configs/dev.yaml configs/production.yaml
notebooks/README.md
notebooks/01_data_understanding_eda.ipynb
notebooks/02_feature_engineering_experiments.ipynb
notebooks/03_model_diagnostics.ipynb
src/regression_intelligence/__init__.py
src/regression_intelligence/data/__init__.py
src/regression_intelligence/data/generate_dataset.py
src/regression_intelligence/data/ingestion.py
src/regression_intelligence/data/validation.py
src/regression_intelligence/data/cleaning.py
src/regression_intelligence/data/splitting.py
src/regression_intelligence/features/__init__.py
src/regression_intelligence/features/engineering.py
src/regression_intelligence/features/preprocessing.py
src/regression_intelligence/models/__init__.py
src/regression_intelligence/models/model_factory.py
src/regression_intelligence/models/training.py
src/regression_intelligence/models/tuning.py
src/regression_intelligence/models/evaluation.py
src/regression_intelligence/models/comparison.py
src/regression_intelligence/analysis/__init__.py
src/regression_intelligence/analysis/coefficients.py
src/regression_intelligence/analysis/residuals.py
src/regression_intelligence/pipelines/__init__.py
src/regression_intelligence/pipelines/training_pipeline.py
src/regression_intelligence/pipelines/prediction_pipeline.py
src/regression_intelligence/api/__init__.py
src/regression_intelligence/api/app.py
src/regression_intelligence/api/schemas.py
src/regression_intelligence/utils/__init__.py
src/regression_intelligence/utils/config.py
src/regression_intelligence/utils/io.py
src/regression_intelligence/utils/logging.py
src/regression_intelligence/utils/paths.py
scripts/run_ingestion.py scripts/run_validation.py scripts/run_training.py
scripts/run_evaluation.py scripts/predict.py
reports/data_dictionary.md reports/assumptions.md reports/model_card.md reports/final_report.md
tests/__init__.py tests/conftest.py tests/test_config.py
tests/test_data_validation.py tests/test_features.py tests/test_preprocessing.py
tests/test_training.py tests/test_prediction_api.py
"""

GITKEEP_DIRS = """
data/raw data/interim data/processed data/external
artifacts/models artifacts/metrics artifacts/predictions
artifacts/metadata artifacts/experiments reports/figures
"""

for f in FILES.split():
    p = Path(f)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch(exist_ok=True)

for d in GITKEEP_DIRS.split():
    Path(d).mkdir(parents=True, exist_ok=True)
    (Path(d) / ".gitkeep").touch(exist_ok=True)

print("Scaffold complete.")
