
import pandas as pd
from pathlib import Path
import xgboost as xgb
import pickle

PAIRS = Path("data/pairs/pairs.csv")
OUT_DIR = Path("data/ranking")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not PAIRS.exists():
        raise SystemExit("pairs.csv not found. Run: python src/utils/build_pair_dataset.py")

    df = pd.read_csv(PAIRS)
    X = df[["user_len", "item_len"]]
    y = df["label"]

    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "binary:logistic", "eval_metric": "auc", "max_depth": 4}
    model = xgb.train(params, dtrain, num_boost_round=50)

    with open(OUT_DIR / "ranker.xgb", "wb") as f:
        pickle.dump(model, f)

    print("Saved XGBoost ranker ->", OUT_DIR / "ranker.xgb")

if __name__ == "__main__":
    main()
