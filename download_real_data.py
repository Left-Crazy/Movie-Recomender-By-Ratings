import io
import ssl
import zipfile
import urllib.request
import pandas as pd

# Bypass SSL certificate verification issues on Windows
ssl_context = ssl._create_unverified_context()

print("Downloading real MovieLens dataset...")
url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Set standard User-Agent header
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

with urllib.request.urlopen(req, context=ssl_context) as resp:
    zf = zipfile.ZipFile(io.BytesIO(resp.read()))

root = zf.namelist()[0].split('/')[0]
movies = pd.read_csv(zf.open(f"{root}/movies.csv"))
ratings = pd.read_csv(zf.open(f"{root}/ratings.csv"))
links = pd.read_csv(zf.open(f"{root}/links.csv"))

# Clean titles and genres
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)\s*$', expand=False).astype('Int64')
movies['clean_title'] = movies['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True).str.strip()
movies['genres'] = movies['genres'].replace('(no genres listed)', '')

links['tmdbId'] = links['tmdbId'].astype('Int64')
movies = movies.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')
movies = movies[['movieId', 'title', 'clean_title', 'year', 'genres', 'tmdbId']]

ratings['rating_date'] = pd.to_datetime(ratings['timestamp'], unit='s').dt.date.astype(str)
ratings = ratings[['userId', 'movieId', 'rating', 'rating_date', 'timestamp']]

movies.to_csv("movies.csv", index=False)
ratings.to_csv("ratings.csv", index=False)
print("Updated movies.csv and ratings.csv with real movies!")