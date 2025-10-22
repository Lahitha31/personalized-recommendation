
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

ARTIFACTS = Path("data/retrieval")
PAIRS_DIR = Path("data/pairs")
PAIRS_DIR.mkdir(parents=True, exist_ok=True)



def main():
    users_path = ARTIFACTS / "user_ids.txt"
    items_path = ARTIFACTS / "movie_titles.txt"

    if not users_path.exists() or not items_path.exists():
        # Fallback: tiny synthetic IDs
        user_ids = [str(i) for i in range(100)]
        item_ids = [str(i) for i in range(200)]
    else:
        user_ids = [l.strip() for l in users_path.read_text().splitlines() if l.strip()]
        item_ids = [l.strip() for l in items_path.read_text().splitlines() if l.strip()]

    rng = np.random.default_rng(7)
    rows = []
    for u in rng.choice(user_ids, size=min(1000, len(user_ids)), replace=True):
        # Sample a few positives and negatives
        pos_items = rng.choice(item_ids, size=3, replace=False)
        neg_items = rng.choice(item_ids, size=7, replace=False)
        for it in pos_items:
            rows.append((u, it, 1))
        for it in neg_items:
            rows.append((u, it, 0))

    df = pd.DataFrame(rows, columns=["user_id", "item_id", "label"])
    # Simple numeric encodings as "features"
    df["user_len"] = df["user_id"].astype(str).str.len()
    df["item_len"] = df["item_id"].astype(str).str.len()

    out_csv = PAIRS_DIR / "pairs.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with shape {df.shape}")

if __name__ == "__main__":
    main()
