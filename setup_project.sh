#!/usr/bin/env bash
# ----------------------------------------------
# FULL PROJECT SETUP 
#
# To create this script:
# cat << 'EOF' > setup_project.sh
# EOF
#
# Usage:
# chmod +x setup_project.sh
# ./setup_project.sh
# ----------------------------------------------

set -euo pipefail

# ----------------------------
# CONFIG
# ----------------------------
PROJECT_NAME="fsi_credit_demand_forecasting"
PYTHON_CMD="python3.10"
POETRY_VERSION="1.8.3"

printf -- "==============================================\n"
printf -- "PROJECT SETUP STARTED\n"
printf -- "Project: %s\n" "$PROJECT_NAME"
printf -- "==============================================\n"

# ----------------------------
# VALIDATION
# ----------------------------
printf -- "--- 1. Checking Python ---\n"
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    printf -- "ERROR: Python 3.10 not found.\n"
    exit 1
fi
printf -- "Python OK: %s\n" "$($PYTHON_CMD --version)"

printf -- "--- 2. Checking Poetry ---\n"
if ! command -v poetry &> /dev/null; then
    printf -- "ERROR: Poetry not installed.\n"
    printf -- "Install with: pipx install poetry==%s\n" "$POETRY_VERSION"
    exit 1
fi
printf -- "Poetry OK: %s\n" "$(poetry --version)"

# ----------------------------
# PROJECT CONTEXT
# ----------------------------
printf -- "--- 3. Using current directory as project root ---\n"

CURRENT_DIR=$(basename "$PWD")

if [ "$CURRENT_DIR" != "$PROJECT_NAME" ]; then
    printf -- "WARNING: Current directory name (%s) is different from project name (%s)\n" "$CURRENT_DIR" "$PROJECT_NAME"
fi

# ----------------------------
# PYPROJECT 
# ----------------------------
printf -- "--- 4. Creating pyproject.toml ---\n"

cat <<EOF > pyproject.toml
[tool.poetry]
name = "fsi_credit_demand_forecasting"
version = "0.1.0"
description = "Forecasting de demanda de crédito com Arquitetura Medalhão e MLOps"
authors = ["Alexandre Rodrigues <alexandre.dsa4@gmail.com>"]
readme = "README.md"
packages = [{ include = "fsi_credit", from = "src" }]

[tool.poetry.dependencies]
python = "==3.10.*"
pandas = "==2.2.2"
numpy = "==1.26.4"
scipy = "==1.13.1"
pyarrow = "==15.0.0"
torch = "==2.3.1"
joblib = "==1.5.3"
fastapi = "^0.110.0"
uvicorn = { extras = ["standard"], version = "^0.27.0" }
pydantic = "^2.6.0"
prefect = { version = "^2.16.0", extras = ["aws"] }
mlflow = { version = "^2.11.0", extras = ["extras"] }
dvc = { version = "^3.48.0", extras = ["s3"] }
hydra-core = "^1.3.2"
omegaconf = "^2.3.0"
pandera = "^0.18.0"
loguru = "^0.7.2"
python-dotenv = "^1.0.1"
ray = "^2.10.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
pytest-cov = "^4.1.0"
black = "^24.2.0"
ruff = "^0.3.0"
isort = "^5.13.2"
pre-commit = "^3.6.0"
pip-audit = "^2.7.0"

[tool.poetry.scripts]
run-api = "fsi_credit.api.main:run"
pipeline-bronze = "fsi_credit.pipelines.ingest_to_bronze:main"
pipeline-silver = "fsi_credit.pipelines.bronze_to_silver:main"
pipeline-gold = "fsi_credit.pipelines.silver_to_gold:main"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 88
target-version = ["py310"]

[tool.isort]
profile = "black"
line_length = 88

[tool.ruff]
line-length = 88
target-version = "py310"
select = ["E","F","W","I"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.coverage.run]
branch = true
source = ["src/fsi_credit"]

[tool.coverage.report]
show_missing = true
EOF

# ----------------------------
# STRUCTURE
# ----------------------------
printf -- "--- 5. Creating structure ---\n"

mkdir -p \
src/fsi_credit/{api,pipelines,models,services,utils,features,validation,evaluation,config,orchestration,jobs} \
tests \
data/{01_bronze,02_silver,03_gold} \
experiments logs artifacts models scripts .github/workflows

touch README.md .env.example .gitignore Dockerfile docker-compose.yaml Makefile
touch src/fsi_credit/__init__.py

# ----------------------------
# BASE FILES
# ----------------------------
printf -- "--- 6. Creating base code ---\n"

cat <<EOF > src/fsi_credit/api/main.py
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="FSI Credit API")

@app.get("/")
def health():
    return {"status": "ok"}

def run():
    uvicorn.run("fsi_credit.api.main:app", host="0.0.0.0", port=8000, reload=True)
EOF

cat <<EOF > src/fsi_credit/pipelines/ingest_to_bronze.py
def main():
    print("Bronze pipeline")
EOF

cat <<EOF > src/fsi_credit/pipelines/bronze_to_silver.py
def main():
    print("Silver pipeline")
EOF

cat <<EOF > src/fsi_credit/pipelines/silver_to_gold.py
def main():
    print("Gold pipeline")
EOF

cat <<EOF > src/fsi_credit/orchestration/flows.py
from prefect import flow
from fsi_credit.pipelines.ingest_to_bronze import main as bronze
from fsi_credit.pipelines.bronze_to_silver import main as silver
from fsi_credit.pipelines.silver_to_gold import main as gold

@flow
def medalion_flow():
    bronze()
    silver()
    gold()
EOF

cat <<EOF > tests/test_basic.py
def test_ok():
    assert True
EOF

# ----------------------------
# GITIGNORE
# ----------------------------
cat <<EOF > .gitignore
.venv/
__pycache__/
*.pyc
.env
data/
logs/
artifacts/
models/
EOF

# ----------------------------
# INSTALL
# ----------------------------
printf -- "--- 7. Installing dependencies ---\n"
poetry config virtualenvs.in-project true
poetry install

# GIT
printf -- "--- 8. Initializing git ---\n"
git init
git add .

if git config user.email &> /dev/null; then
    git commit -m "Initial commit"
else
    printf -- "WARNING: Git not configured. Skipping commit.\n"
fi

# DVC
printf -- "--- 9. Initializing DVC ---\n"
poetry run dvc init

# ----------------------------
# DONE
# ----------------------------
printf -- "\n==============================================\n"
printf -- "SETUP COMPLETED SUCCESSFULLY\n"
printf -- "==============================================\n"

printf -- "Next steps:\n"
printf -- "Current directory is ready\n"
printf -- "poetry run run-api\n"

printf -- "\n----------------------------------------------\n"
printf -- "Environment Validation Checklist\n"
printf -- "----------------------------------------------\n"
printf -- "1. Check Python version:\n"
printf -- "   poetry run python -V\n\n"

printf -- "2. Check dependencies:\n"
printf -- "   poetry check\n\n"

printf -- "3. Run tests:\n"
printf -- "   poetry run pytest\n\n"

printf -- "4. Start API:\n"
printf -- "   poetry run run-api\n\n"

printf -- "5. Validate API (browser or curl):\n"
printf -- "   http://127.0.0.1:8000\n"
printf -- "   curl http://127.0.0.1:8000\n"

printf -- "----------------------------------------------\n"