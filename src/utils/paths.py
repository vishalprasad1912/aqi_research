from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

CONFIG_DIR = PROJECT_DIR / "config"
SPLITS_DIR = PROJECT_DIR / "data" / "splits"
MODELS_SAVED_DIR = PROJECT_DIR / "models_saved"
RESULTS_DIR = PROJECT_DIR / "results"

VISUALIZATION_DIR = SRC_DIR / "visualization"
EXTERNAL_DIR = PROJECT_DIR / "data" / "external"
