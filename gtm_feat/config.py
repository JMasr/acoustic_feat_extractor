from os import makedirs
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
MODELS_PRETRAINED_DIR = MODELS_DIR / "pretrained"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = REPORTS_DIR / "logs"
TEST_DIR = PROJ_ROOT / "tests"

# Check all project directories exist
project_directories: list[Path] = [
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    MODELS_DIR,
    MODELS_PRETRAINED_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    TEST_DIR
]
for directory in project_directories:
    makedirs(directory, exist_ok=True)

# Environment variables
DOT_ENV_FILE = PROJ_ROOT / ".env"
load_dotenv(DOT_ENV_FILE)

logger.add(
    LOGS_DIR / "gtm_feat.log",
    rotation="1 week",
    retention="1 month",
    level="DEBUG",
)

logger.info("Configuration loaded successfully.")
