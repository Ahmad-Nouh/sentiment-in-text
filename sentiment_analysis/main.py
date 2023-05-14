import os
import json
import pandas as pd
import warnings
from argparse import Namespace
from pathlib import Path
import config.config as config
from sentiment_analysis import data, train, utils

warnings.filterwarnings("ignore")

def lt_data():
    """load and transform our data assets."""
    df = pd.read_csv(os.path.join(config.RAW_DATA_DIR, "text_emotion.csv"), index_col=False)
    df.dropna(inplace=True)
    return df

def train_model(args_fp):
    """Train a model given arguments."""
    # Load labeled data
    df = pd.read_csv(Path(config.RAW_DATA_DIR / "text_emotion.csv"), index_col=False)

    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    artifacts = train.train(df=df, args=args)
    performance = artifacts["performance"]
    print(json.dumps(performance, indent=2))


if __name__ == "__main__":
    args_fp = os.path.join(config.CONFIG_DIR, "args.json")
    train_model(args_fp)