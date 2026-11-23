"""Download the Titanic dataset and write an 80/20 train/test split to data/.

Source: seaborn-data (public, MIT-style sample data used in many tutorials).
Run once:  python src/prepare_data.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COLUMNS = ["pclass", "sex", "age", "fare", "survived"]  # 4 features + target


def main() -> None:
    df = pd.read_csv(URL)[COLUMNS]
    df = df.dropna(subset=["fare", "survived"])  # age may stay missing: the model imputes it

    # stratify keeps the survived/died ratio equal in both files.
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["survived"]
    )
    DATA_DIR.mkdir(exist_ok=True)
    train.to_csv(DATA_DIR / "train.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)
    print(f"train: {len(train)} rows, test: {len(test)} rows -> {DATA_DIR}")


if __name__ == "__main__":
    main()
