"""
dataset.py
----------
Acquisition + cleaning script for the Movie Recommendation Engine.

Downloads the official MovieLens "ml-latest-small" dataset directly from
GroupLens (https://grouplens.org/datasets/movielens/) -- 9,742 movies and
100,836 ratings from 610 users -- and turns it into the two clean files
the rest of the project reads:

    movies.csv   movieId, title, clean_title, year, genres, tmdbId
    ratings.csv  userId, movieId, rating, rating_date, timestamp

`movies.csv` and `ratings.csv` are already included in this project, so
you do NOT need to run this script to use the app. Run it only if you
want to re-download a fresh copy of the source data or swap in a larger
MovieLens release (see MOVIELENS_URL below).

Usage:
    python dataset.py
"""

import io
import re
import zipfile
import urllib.request

import pandas as pd

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
# For a bigger dataset (more movies, more ratings), swap in:
#   https://files.grouplens.org/datasets/movielens/ml-latest.zip   (~87,000 movies)
# It's a much larger download and will make the TF-IDF / SVD steps slower.


def download_movielens(url=MOVIELENS_URL):
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(data))
    root = zf.namelist()[0].split("/")[0]
    movies = pd.read_csv(zf.open(f"{root}/movies.csv"))
    ratings = pd.read_csv(zf.open(f"{root}/ratings.csv"))
    links = pd.read_csv(zf.open(f"{root}/links.csv"))
    print(f"Downloaded {len(movies)} movies, {len(ratings)} ratings.")
    return movies, ratings, links


def extract_year(title):
    m = re.search(r"\((\d{4})\)\s*$", title)
    return int(m.group(1)) if m else pd.NA


def clean(movies, ratings, links):
    movies = movies.copy()
    movies["year"] = movies["title"].apply(extract_year).astype("Int64")
    movies["clean_title"] = (
        movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()
    )
    # "(no genres listed)" is a placeholder, not a real genre -- drop it so it
    # doesn't pollute the TF-IDF vocabulary used by the content-based engine.
    movies["genres"] = movies["genres"].replace("(no genres listed)", "")

    links = links.copy()
    links["tmdbId"] = links["tmdbId"].astype("Int64")
    movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    movies = movies[["movieId", "title", "clean_title", "year", "genres", "tmdbId"]]

    ratings = ratings.drop_duplicates(subset=["userId", "movieId"]).copy()
    ratings["rating_date"] = pd.to_datetime(ratings["timestamp"], unit="s").dt.date.astype(str)
    ratings = ratings[["userId", "movieId", "rating", "rating_date", "timestamp"]]

    return movies, ratings


def main():
    raw_movies, raw_ratings, links = download_movielens()
    movies, ratings = clean(raw_movies, raw_ratings, links)
    movies.to_csv("movies.csv", index=False)
    ratings.to_csv("ratings.csv", index=False)
    print(f"Wrote movies.csv ({len(movies)} rows) and ratings.csv ({len(ratings)} rows).")


if __name__ == "__main__":
    main()
