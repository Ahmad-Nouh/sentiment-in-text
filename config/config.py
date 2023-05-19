from pathlib import Path
import mlflow

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")

RAW_DATA_DIR = Path(DATA_DIR, "raw")
PROCESSED_DATA_DIR = Path(DATA_DIR, "processed")

STORES_DIR = Path(BASE_DIR, "stores")
MODEL_REGISTRY = Path(STORES_DIR, "model")

# Create dirs
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)

EMOJI_DICT_FILE = str(Path(RAW_DATA_DIR, "Emoji_Dict.p"))

with open(str(Path(RAW_DATA_DIR, "stopwords.txt")), "r") as file:
    STOPWORDS = [line.strip() for line in file.readlines()]



mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))