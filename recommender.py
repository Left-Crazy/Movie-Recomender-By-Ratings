"""
recommender.py
--------------
Backend Movie Recommendation Engine supporting both static MovieLens users
and real-time session ratings (fold-in SVD projection).
"""

import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD


class MovieRecommender:
    def __init__(self, movies_path="movies.csv", ratings_path="ratings.csv",
                 svd_components=50, random_state=42):
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)

        self.movies["genres"] = self.movies["genres"].fillna("")
        self.movies["clean_title"] = self.movies["clean_title"].fillna(self.movies["title"])
        self.movies = self.movies.reset_index(drop=True)

        self.id_to_idx = {mid: i for i, mid in enumerate(self.movies["movieId"])}
        self.idx_to_id = {i: mid for mid, i in self.id_to_idx.items()}
        self.title_to_id = dict(zip(self.movies["clean_title"].str.lower(), self.movies["movieId"]))

        self._build_content_model()
        self._build_collaborative_model(svd_components, random_state)
        self._build_popularity_table()

    # ---------------------------------------------------------------
    # 1. Content-Based Filtering
    # ---------------------------------------------------------------
    def _build_content_model(self):
        soup = (
            (self.movies["clean_title"] + " ") * 2
            + self.movies["genres"].str.replace("|", " ", regex=False)
        )
        self.tfidf = TfidfVectorizer(stop_words="english", min_df=1)
        self.tfidf_matrix = self.tfidf.fit_transform(soup)

    def _similar_by_index(self, idx, top_n, exclude=frozenset()):
        query = self.tfidf_matrix[idx]
        sims = linear_kernel(query, self.tfidf_matrix).ravel()
        order = np.argsort(-sims)
        picks = []
        for i in order:
            if i == idx or i in exclude:
                continue
            picks.append((i, sims[i]))
            if len(picks) == top_n:
                break
        return picks

    def content_based_recommend(self, title, top_n=10):
        mid = self._resolve_title(title)
        if mid is None:
            return pd.DataFrame()
        idx = self.id_to_idx[mid]
        picks = self._similar_by_index(idx, top_n)
        return self._format_results(picks)

    def content_based_for_user(self, user_id, top_n=10, like_threshold=4.0):
        user_ratings = self.ratings[self.ratings.userId == user_id]
        if user_ratings.empty:
            return self.popular_movies(top_n)

        liked = user_ratings[user_ratings.rating >= like_threshold]
        if liked.empty:
            liked = user_ratings.nlargest(5, "rating")

        liked_idx = [self.id_to_idx[m] for m in liked.movieId if m in self.id_to_idx]
        if not liked_idx:
            return self.popular_movies(top_n)

        profile = np.asarray(self.tfidf_matrix[liked_idx].mean(axis=0))
        sims = linear_kernel(profile, self.tfidf_matrix).ravel()

        seen = {self.id_to_idx[m] for m in user_ratings.movieId if m in self.id_to_idx}
        order = np.argsort(-sims)
        picks = [(i, sims[i]) for i in order if i not in seen][:top_n]
        return self._format_results(picks)

    def content_based_from_session(self, session_ratings, top_n=10, like_threshold=3.5):
        """Content-based filtering for real-time live ratings."""
        liked_mids = [m for m, r in session_ratings.items() if r >= like_threshold and m in self.id_to_idx]
        if not liked_mids:
            liked_mids = list(session_ratings.keys())
        if not liked_mids:
            return self.popular_movies(top_n)

        liked_idx = [self.id_to_idx[m] for m in liked_mids]
        profile = np.asarray(self.tfidf_matrix[liked_idx].mean(axis=0))
        sims = linear_kernel(profile, self.tfidf_matrix).ravel()

        seen = {self.id_to_idx[m] for m in session_ratings.keys() if m in self.id_to_idx}
        order = np.argsort(-sims)
        picks = [(i, sims[i]) for i in order if i not in seen][:top_n]
        return self._format_results(picks)

    # ---------------------------------------------------------------
    # 2. Collaborative Filtering (Truncated SVD)
    # ---------------------------------------------------------------
    def _build_collaborative_model(self, k, random_state):
        user_ids = sorted(self.ratings["userId"].unique())
        self.user_to_idx = {u: i for i, u in enumerate(user_ids)}

        n_users = len(self.user_to_idx)
        n_movies = len(self.movies)

        rows = self.ratings["userId"].map(self.user_to_idx).to_numpy()
        cols = self.ratings["movieId"].map(self.id_to_idx).to_numpy()
        vals = self.ratings["rating"].to_numpy(dtype=float)
        R = csr_matrix((vals, (rows, cols)), shape=(n_users, n_movies))

        counts = np.diff(R.indptr)
        counts_safe = np.where(counts == 0, 1, counts)
        sums = np.asarray(R.sum(axis=1)).ravel()
        self.user_means = sums / counts_safe

        R_centered = R.copy().astype(float)
        for u in range(n_users):
            start, end = R.indptr[u], R.indptr[u + 1]
            R_centered.data[start:end] -= self.user_means[u]

        k = max(2, min(k, min(R.shape) - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=random_state)
        self.U_sigma = self.svd.fit_transform(R_centered)
        self.Vt = self.svd.components_  # Shape: (k, n_movies)

    def predict_ratings_for_user(self, user_id):
        if user_id not in self.user_to_idx:
            return None
        u = self.user_to_idx[user_id]
        preds = self.U_sigma[u] @ self.Vt + self.user_means[u]
        return np.clip(preds, 0.5, 5.0)

    def predict_ratings_from_session(self, session_ratings):
        """Real-time SVD fold-in projection for active live ratings."""
        if not session_ratings:
            return None

        n_movies = len(self.movies)
        r = np.zeros(n_movies)
        for mid, rating in session_ratings.items():
            if mid in self.id_to_idx:
                r[self.id_to_idx[mid]] = rating

        user_mean = np.mean(list(session_ratings.values()))
        r_centered = np.zeros(n_movies)
        for mid, rating in session_ratings.items():
            if mid in self.id_to_idx:
                r_centered[self.id_to_idx[mid]] = rating - user_mean

        # Linear Projection into latent space: q = r_centered @ V
        # where V = self.Vt.T
        latent_profile = r_centered @ self.Vt.T  # Shape: (k,)
        preds = latent_profile @ self.Vt + user_mean
        return np.clip(preds, 0.5, 5.0)

    def collaborative_recommend(self, user_id, top_n=10):
        preds = self.predict_ratings_for_user(user_id)
        if preds is None:
            return self.popular_movies(top_n)

        seen_ids = self.ratings.loc[self.ratings.userId == user_id, "movieId"]
        seen = {self.id_to_idx[m] for m in seen_ids if m in self.id_to_idx}

        order = np.argsort(-preds)
        picks = [(i, preds[i]) for i in order if i not in seen][:top_n]
        return self._format_results(picks, score_col="predicted_rating")

    def collaborative_from_session(self, session_ratings, top_n=10):
        preds = self.predict_ratings_from_session(session_ratings)
        if preds is None:
            return self.popular_movies(top_n)

        seen = {self.id_to_idx[m] for m in session_ratings.keys() if m in self.id_to_idx}
        order = np.argsort(-preds)
        picks = [(i, preds[i]) for i in order if i not in seen][:top_n]
        return self._format_results(picks, score_col="predicted_rating")

    # ---------------------------------------------------------------
    # 3. Hybrid Scoring
    # ---------------------------------------------------------------
    def hybrid_recommend(self, user_id, title=None, top_n=10, alpha=0.5, pool_size=50):
        if title:
            mid = self._resolve_title(title)
            pool = (self._format_results(self._similar_by_index(self.id_to_idx[mid], pool_size))
                    if mid is not None else self.content_based_for_user(user_id, pool_size))
        else:
            pool = self.content_based_for_user(user_id, pool_size)

        if pool.empty:
            return self.popular_movies(top_n)

        preds = self.predict_ratings_for_user(user_id)
        if preds is None:
            return pool.head(top_n)

        return self._rank_pool(pool, preds, alpha, top_n)

    def hybrid_from_session(self, session_ratings, top_n=10, alpha=0.5, pool_size=50):
        """Real-time hybrid recommendations from interactive session ratings."""
        pool = self.content_based_from_session(session_ratings, top_n=pool_size)
        if pool.empty:
            return self.popular_movies(top_n)

        preds = self.predict_ratings_from_session(session_ratings)
        if preds is None:
            return pool.head(top_n)

        return self._rank_pool(pool, preds, alpha, top_n)

    def _rank_pool(self, pool, preds, alpha, top_n):
        pool = pool.copy()
        pool["predicted_rating"] = pool["movieId"].map(
            lambda m: preds[self.id_to_idx[m]] if m in self.id_to_idx else np.nan
        )

        def norm(s):
            lo, hi = s.min(), s.max()
            return (s - lo) / (hi - lo) if hi > lo else pd.Series(1.0, index=s.index)

        content_col = "similarity" if "similarity" in pool.columns else None
        pool["content_norm"] = norm(pool[content_col]) if content_col else 1.0
        pool["cf_norm"] = norm(pool["predicted_rating"])
        pool["hybrid_score"] = alpha * pool["content_norm"] + (1 - alpha) * pool["cf_norm"]

        return pool.sort_values("hybrid_score", ascending=False).head(top_n).reset_index(drop=True)

    # ---------------------------------------------------------------
    # 4. Popularity & Helpers
    # ---------------------------------------------------------------
    def _build_popularity_table(self):
        stats = self.ratings.groupby("movieId")["rating"].agg(["mean", "count"])
        C = stats["mean"].mean()
        m = stats["count"].quantile(0.75)
        stats["weighted"] = ((stats["count"] / (stats["count"] + m)) * stats["mean"]
                              + (m / (stats["count"] + m)) * C)
        self.popularity = stats.sort_values("weighted", ascending=False)

    def popular_movies(self, top_n=10, genre=None):
        pool = self.popularity
        if genre:
            in_genre = self.movies.loc[self.movies["genres"].str.contains(genre, na=False), "movieId"]
            pool = pool.loc[pool.index.isin(in_genre)]
        top_ids = pool.head(top_n).index
        result = self.movies.set_index("movieId").loc[top_ids].reset_index()
        result["score"] = pool.loc[top_ids, "weighted"].values
        return result

    def _resolve_title(self, query):
        matches = self.search_titles(query, limit=1)
        return matches[0]["movieId"] if matches else None

    def search_titles(self, query, limit=10):
        import re
        q = str(query).strip().lower()
        if not q:
            return []

        # Tokenize query words
        q_clean = re.sub(r'[^a-z0-9\s]', ' ', q)
        q_tokens = q_clean.split()
        q_stripped = "".join(q_tokens)

        if not q_tokens:
            return []

        results = []
        for idx, row in self.movies.iterrows():
            title_clean = str(row.get("clean_title", row.get("title", ""))).lower()
            t_clean = re.sub(r'[^a-z0-9\s]', ' ', title_clean)
            t_stripped = "".join(t_clean.split())

            # Strategy 1: Continuous stripped match (e.g. "spidermanhomecoming")
            if q_stripped in t_stripped:
                results.append(row)
                continue

            # Strategy 2: All individual query words exist in the title
            if all(token in t_clean for token in q_tokens):
                results.append(row)
                continue

        if not results:
            return []

        df_matches = pd.DataFrame(results).drop_duplicates(subset=["movieId"])
        return df_matches[["movieId", "title"]].head(limit).to_dict("records")

    def get_user_ids(self):
        return list(self.user_to_idx.keys())

    def get_all_genres(self):
        genres = set()
        for g in self.movies["genres"]:
            if g:
                genres.update(g.split("|"))
        return sorted(genres)

    def _format_results(self, pairs, score_col="similarity"):
        rows = []
        for i, score in pairs:
            m = self.movies.iloc[i]
            rows.append({
                "movieId": int(m["movieId"]),
                "title": m["title"],
                "genres": m["genres"],
                "year": None if pd.isna(m["year"]) else int(m["year"]),
                "tmdbId": None if pd.isna(m["tmdbId"]) else int(m["tmdbId"]),
                score_col: round(float(score), 4),
            })
        return pd.DataFrame(rows)