
# Personalized Recommendation System (Two-Stage: Retrieval + Ranking)



- **Two-tower retrieval** (TensorFlow Recommenders) to fetch a small candidate set quickly.
- **Learning-to-rank** (XGBoost) to re-score candidates for **personalization & diversity**.
- **Metric:** track **NDCG@10** (baseline included).
- **Serving:** FastAPI microservice with a `/recommend` endpoint.
- **Ops:** Dockerfile and GitHub Actions CI with linting, tests, and security checks.
- **Feature Store (optional):** Feast skeleton with local file source so you can extend to online/offline stores later.

> This repo trains on MovieLens 100k (via `tensorflow_datasets`) so it's reproducible on any laptop.
> The **ranking** stage is wired to use simple features out-of-the-box. You can upgrade to richer Feast features later.

---

## Quickstart

```bash
# 1) Create venv
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip

# 2) Install deps
pip install -r requirements.txt

# 3) Train retrieval model (TFRS, MovieLens-100k)
python src/retrieval/train_retrieval.py --epochs 3 --k 100

# 4) Export training pairs + simple features (from retrieval artifacts)
python src/utils/build_pair_dataset.py

# 5) Train ranking model (XGBoost) on exported pairs
python src/ranking/train_ranking.py

# 6) Evaluate NDCG@10
python src/evaluate/ndcg.py --k 10

# 7) Serve API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 8) Try it
curl "http://localhost:8000/recommend?user_id=42&k=10"

```

Artifacts are written to `data/`:
- `data/retrieval/` — SavedModel, vocabularies, candidate index
- `data/pairs/` — Training pairs for the ranking stage
- `data/ranking/` — XGBoost model + encoders
- `data/metrics/` — evaluation JSON

---

## Docker

```bash
docker build -t personalized-recs:latest .
docker run -p 8000:8000 personalized-recs:latest
```

---

## CI/CD 

- Lint & format (ruff, black)
- Type check (mypy)
- Tests (pytest)
- Security checks (bandit, safety)
- **SonarCloud** and **Veracode** steps are included as commented templates. Un-comment & set secrets to enable.

---


