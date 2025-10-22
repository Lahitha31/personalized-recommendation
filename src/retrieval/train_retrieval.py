
import argparse
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# Minimal two-tower on MovieLens-100k
# Saves artifacts under data/retrieval/

ART_DIR = Path("data/retrieval")
ART_DIR.mkdir(parents=True, exist_ok=True)

def prepare_datasets():
    ratings = tfds.load("movielens/100k-ratings", split="train")
    ratings = ratings.map(lambda x: {
        "user_id": x["user_id"],
        "movie_title": x["movie_title"],
    })

    users = ratings.map(lambda x: x["user_id"])
    movies = ratings.map(lambda x: x["movie_title"])

    user_ids = np.unique(np.concatenate(list(users.batch(1000))))
    movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))

    np.savetxt(ART_DIR / "user_ids.txt", user_ids, fmt="%s")
    np.savetxt(ART_DIR / "movie_titles.txt", movie_titles, fmt="%s")

    vocab_users = tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None)
    vocab_movies = tf.keras.layers.StringLookup(vocabulary=movie_titles, mask_token=None)

    # Shuffle & split
    tf.random.set_seed(42)
    ratings = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
    n = 80_000
    train = ratings.take(n)
    test = ratings.skip(n).take(20_000)
    return train, test, vocab_users, vocab_movies, user_ids, movie_titles

class TwoTowerModel(tfrs.models.Model):
    def __init__(self, user_model, item_model, task):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.task = task

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user_id"])
        item_embeddings = self.item_model(features["movie_title"])
        return self.task(user_embeddings, item_embeddings)

def build_model(vocab_users, vocab_movies, embedding_dim=32):
    user_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=vocab_users.get_vocabulary(), mask_token=None),
        tf.keras.layers.Embedding(vocab_users.vocabulary_size(), embedding_dim),
    ])
    item_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=vocab_movies.get_vocabulary(), mask_token=None),
        tf.keras.layers.Embedding(vocab_movies.vocabulary_size(), embedding_dim),
    ])

    # Retrieval task with factorized top-k
    candidate_dataset = tf.data.Dataset.from_tensor_slices(vocab_movies.get_vocabulary()).map(lambda x: {"movie_title": x})
    metrics = tfrs.metrics.FactorizedTopK(
        candidates=candidate_dataset.batch(128).map(lambda x: item_model(x["movie_title"]))
    )
    task = tfrs.tasks.Retrieval(metrics=metrics)

    model = TwoTowerModel(user_model, item_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args()

    train, test, vocab_users, vocab_movies, user_ids, movie_titles = prepare_datasets()
    cached_train = train.batch(8192).cache()
    cached_test = test.batch(4096).cache()

    model = build_model(vocab_users, vocab_movies)
    model.fit(cached_train, epochs=args.epochs, verbose=2)

    eval_res = model.evaluate(cached_test, return_dict=True)
    print("Retrieval eval:", eval_res)

    # Export user/item towers
    tf.saved_model.save(model.user_model, str(ART_DIR / "user_tower"))
    tf.saved_model.save(model.item_model, str(ART_DIR / "item_tower"))

    # Build brute-force top-K index for quick demo serving
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    titles = tf.constant(vocab_movies.get_vocabulary())
    index.index_from_dataset(
    tf.data.Dataset.from_tensor_slices(titles).batch(128).map(lambda x: (x, model.item_model(x))))

    tf.saved_model.save(index, str(ART_DIR / "bruteforce_index"))

    # Save minimal eval
    (ART_DIR / "metrics.json").write_text(str(eval_res))

    print("Saved artifacts to", ART_DIR)

if __name__ == "__main__":
    main()
