from fastapi import FastAPI, Query
from pathlib import Path
from fastapi.responses import HTMLResponse
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import xgboost as xgb


app = FastAPI(title="Personalized Recs API", version="0.1.2")

RETR_DIR = Path("data/retrieval")
RANK_DIR = Path("data/ranking")


def _clean_b_prefix(s: str) -> str:
    """
    MovieLens titles written via numpy can end up as byte-string reprs like b'Foo'.
    This strips the leading b'' if present, otherwise returns the original string.
    """
    s = s.strip()
    if (s.startswith("b'") and s.endswith("'")) or (s.startswith('b"') and s.endswith('"')):
        return s[2:-1]
    return s


# Build retrieval index at startup
index = None
try:
    user_tower_path = RETR_DIR / "user_tower"
    item_tower_path = RETR_DIR / "item_tower"
    titles_txt = RETR_DIR / "movie_titles.txt"

    if user_tower_path.exists() and item_tower_path.exists() and titles_txt.exists():
        user_tower = tf.saved_model.load(str(user_tower_path))
        item_tower = tf.saved_model.load(str(item_tower_path))

        # Load & clean candidate IDs (movie titles)
        raw_lines = [ln for ln in titles_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        titles = [_clean_b_prefix(ln) for ln in raw_lines]
        titles_tf = tf.constant(titles)

        bf = tfrs.layers.factorized_top_k.BruteForce(user_tower)
        # NOTE: In this TFRS version, use index_from_dataset (not index_exact)
        bf.index_from_dataset(
            tf.data.Dataset.from_tensor_slices(titles_tf)
            .batch(128)
            .map(lambda x: (x, item_tower(x)))
        )
        index = bf
        print("✅ Retrieval index loaded with cleaned titles.")
    else:
        print("⚠️ Retrieval artifacts missing. Train retrieval first.")
except Exception as e:
    print("❌ Failed to build retrieval index:", e)


# Load ranker
ranker = None
try:
    ranker_path = RANK_DIR / "ranker.xgb"
    if ranker_path.exists():
        with open(ranker_path, "rb") as f:
            ranker = pickle.load(f)
        print("✅ Ranker loaded.")
    else:
        print("⚠️ Ranker not found. API will run retrieval-only.")
except Exception as e:
    print("❌ Failed to load ranker:", e)


@app.get("/health")
def health():
    return {
        "ok": True,
        "retrieval_loaded": index is not None,
        "ranker_loaded": ranker is not None,
    }


@app.get("/recommend")
def recommend(user_id: str = Query(...), k: int = Query(10, ge=1, le=100)):
    if index is None:
        return {"error": "Retrieval index not found. Run retrieval training first."}

    # BruteForce returns (scores, ids) in this order for TFRS 0.7.0
    scores_tf, titles_tf = index(tf.constant([user_id]), k=k)

    # Decode titles safely (may already be str)
    raw_titles = titles_tf[0].numpy().tolist()
    titles = [t.decode("utf-8") if isinstance(t, (bytes, np.bytes_)) else str(t) for t in raw_titles]

    scores = [float(s) for s in scores_tf[0].numpy().tolist()]

    # Optional re-ranking with a simple feature set
    if ranker is not None:
        import pandas as pd

        X = pd.DataFrame(
            {
                "user_len": [len(user_id)] * len(titles),
                "item_len": [len(t) for t in titles],
            }
        )
        dmat = xgb.DMatrix(X)
        rank_scores = ranker.predict(dmat).tolist()

        # Blend retrieval score + ranking score
        alpha = 0.5
        final = [(t, alpha * r + (1 - alpha) * s) for t, r, s in zip(titles, rank_scores, scores)]
    else:
        final = list(zip(titles, scores))

    # Sort and return top-k
    final = sorted(final, key=lambda x: x[1], reverse=True)[:k]

    return {
    "user_id": user_id,
    "items": [t for t, sc in final]  # Only return item names, no score
}

@app.get("/", response_class=HTMLResponse)
def demo():
    return """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Personalized Recommendations</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
    :root {
    --bg: #0b1020;
    --panel: #121a33;
    --accent: #4f8cff;
    --text: #e8eeff;
    --muted: #a8b3d6;
    --chip: #1b264a;
    --ok: #2ecc71;
    --warn: #ffb347;
    }
    * { box-sizing: border-box; }
    body {
    margin: 0; font-family: ui-sans-serif, system-ui, Segoe UI, Roboto, Arial;
    background: radial-gradient(1400px 700px at 20% -10%, #0e1733 10%, var(--bg) 60%);
    color: var(--text);
    }
    header {
    padding: 28px 20px; display: grid; gap: 8px; place-items: center;
    }
    h1 {
    margin: 0; font-weight: 800; letter-spacing: 0.3px;
    background: linear-gradient(90deg, #c7d2fe, #93c5fd);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    p.sub { margin: 0; color: var(--muted); font-size: 14px }
    .card {
    width: min(920px, 92vw); margin: 18px auto; background: var(--panel);
    border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    form {
    display: grid; gap: 12px; grid-template-columns: 1fr auto 140px;
    align-items: end;
    }
    label { font-size: 12px; color: var(--muted); }
    input, select {
    width: 100%; padding: 10px 12px; border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.14); background: #0f1630; color: var(--text);
    outline: none;
    }
    button {
    padding: 11px 14px; border-radius: 10px; border: 1px solid #6ea2ff42;
    background: linear-gradient(180deg, #699aff, #4678f7);
    color: white; font-weight: 600; cursor: pointer;
    transition: transform .05s ease-in-out, box-shadow .2s;
    box-shadow: 0 8px 20px rgba(79,140,255,.3);
    }
    button:hover { transform: translateY(-1px); }
    .status { font-size: 12px; color: var(--muted); margin-top: 6px; min-height: 18px; }
    .results {
    display: grid; gap: 10px; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    margin-top: 14px;
    }
    .chip {
    background: var(--chip); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 12px; line-height: 1.3;
    }
    .muted { color: var(--muted); font-size: 13px; }
    .row { display: grid; gap: 12px; grid-template-columns: 1fr 120px; }
    .health { font-size: 12px; margin-top: 8px; }
    .ok { color: var(--ok); } .bad { color: #ff6b6b; } .warn { color: var(--warn); }
    footer { text-align: center; color: var(--muted); font-size: 12px; padding: 18px; }
    .kbd {
    border: 1px solid rgba(255,255,255,0.15); padding: 2px 6px; border-radius: 6px; background: #0f1630;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size: 12px;
    }
</style>
</head>
<body>
<header>
    <h1>Personalized Recommendations</h1>
    <p class="sub">Two-stage retrieval + ranking • FastAPI • TensorFlow Recommenders • XGBoost</p>
</header>

<section class="card">
    <div class="row">
    <div>
        <label>User ID</label>
        <input id="user_id" placeholder="e.g. 42 or a value from user_ids.txt" />
    </div>
    <div>
        <label>Top-K</label>
        <select id="k">
        <option>5</option><option selected>10</option><option>20</option><option>50</option>
        </select>
    </div>
    </div>
    <div class="status" id="status">Tip: You can paste a real user from <span class="kbd">data/retrieval/user_ids.txt</span>.</div>
    <div>
    <form id="form" onsubmit="return false;">
        <div></div>
        <div></div>
        <button id="btn">Get Recommendations</button>
    </form>
    </div>
    <div class="health" id="health"></div>
    <div class="results" id="results"></div>
</section>


<script>
async function checkHealth() {
try {
    const r = await fetch('/health');
    const j = await r.json();
    const el = document.getElementById('health');
    const retr = j.retrieval_loaded ? '<span class="ok">retrieval ✓</span>' : '<span class="bad">retrieval ×</span>';
    const rank = j.ranker_loaded ? '<span class="ok">ranker ✓</span>' : '<span class="warn">ranker −</span>';
    el.innerHTML = 'Status: ' + retr + ' • ' + rank;
} catch (e) {
    document.getElementById('health').innerHTML = '<span class="bad">API unreachable</span>';
}
}
checkHealth();

document.getElementById('btn').addEventListener('click', async () => {
const user = document.getElementById('user_id').value.trim();
const k = document.getElementById('k').value;
const status = document.getElementById('status');
const results = document.getElementById('results');
results.innerHTML = '';
if (!user) {
    status.innerHTML = 'Please enter a User ID.';
    return;
}
status.innerHTML = 'Fetching…';
try {
    const r = await fetch(`/recommend?user_id=${encodeURIComponent(user)}&k=${encodeURIComponent(k)}`);
    const j = await r.json();
    if (j.error) {
    status.innerHTML = '<span class="bad">' + j.error + '</span>';
    return;
    }
    status.innerHTML = 'Got ' + j.items.length + ' items.';
    results.innerHTML = j.items.map(t => `<div class="chip">${t}</div>`).join('');
} catch (e) {
    status.innerHTML = '<span class="bad">Failed to fetch recommendations.</span>';
}
});
</script>
</body>
</html>
    """

