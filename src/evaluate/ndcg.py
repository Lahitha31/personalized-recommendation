
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

METRICS_DIR = Path("data/metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def dcg(rels):
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))

def ndcg_at_k(true_rels, k=10):
    rels = true_rels[:k]
    ideal = sorted(true_rels, reverse=True)[:k]
    return dcg(rels) / (dcg(ideal) + 1e-9)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    # Simple synthetic evaluation: random relevance with slight boost
    rng = np.random.default_rng(0)
    rels = rng.integers(0, 2, size=50).tolist()
    score = ndcg_at_k(rels, k=args.k)

    out = {"ndcg@{}".format(args.k): score}
    (METRICS_DIR / "ndcg.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
