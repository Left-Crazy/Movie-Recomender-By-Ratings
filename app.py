"""
app.py
------
Streamlit UI supporting static user recommendations and real-time live rating sliders.
"""

import pandas as pd
import requests
import streamlit as st

from recommender import MovieRecommender

st.set_page_config(page_title="Movie Recommendation Engine", page_icon="🎬", layout="wide")

# Initialize Session State for Live Interactive Ratings
if "live_ratings" not in st.session_state:
    st.session_state["live_ratings"] = {}


@st.cache_resource(show_spinner="Loading movie data and training SVD model...")
def load_engine():
    return MovieRecommender("movies.csv", "ratings.csv")


@st.cache_data(show_spinner=False)
def fetch_poster(tmdb_id, api_key):
    if not api_key or tmdb_id is None or pd.isna(tmdb_id):
        return None
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}",
            params={"api_key": api_key},
            timeout=5,
        )
        r.raise_for_status()
        path = r.json().get("poster_path")
        return f"https://image.tmdb.org/t/p/w342{path}" if path else None
    except Exception:
        return None


def render_results(results, tmdb_key):
    score_col = next(
        (c for c in ["hybrid_score", "predicted_rating", "similarity", "score"] if c in results.columns),
        None,
    )
    per_row = 5
    for start in range(0, len(results), per_row):
        chunk = results.iloc[start:start + per_row]
        cols = st.columns(len(chunk))
        for col, (_, movie) in zip(cols, chunk.iterrows()):
            with col:
                poster = fetch_poster(movie.get("tmdbId"), tmdb_key) if tmdb_key else None
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.markdown(
                        f"<div style='display:flex;align-items:center;justify-content:center;"
                        f"height:180px;border-radius:8px;background:#22272e;color:#8b98a5;"
                        f"font-size:2rem;font-weight:600;'>{movie['title'][0]}</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown(f"**{movie['title']}**")
                genre_line = movie["genres"].replace("|", " · ") if movie["genres"] else "—"
                st.caption(genre_line)
                if score_col:
                    label = score_col.replace("_", " ").title()
                    st.caption(f"{label}: {movie[score_col]:.2f}")

    with st.expander("View as raw data table"):
        st.dataframe(results, use_container_width=True, hide_index=True)


def main():
    engine = load_engine()

    st.title("🎬 Movie Recommendation Engine")
    st.caption(
        f"Hybrid Model (Content TF-IDF + Real-Time SVD Matrix Factorization) trained on "
        f"{len(engine.movies):,} movies and {len(engine.ratings):,} ratings."
    )

    tab_interactive, tab_preset, tab_browse = st.tabs(
        ["⚡ Live Interactive Ratings (Demo)", "👤 Pre-existing Users", "🔥 Popular by Genre"]
    )

    tmdb_key = st.sidebar.text_input(
        "TMDB API key (optional for posters)", type="password",
        help="Enter key to load live movie posters.",
    )

    # -----------------------------------------------------------------
    # TAB 1: Real-Time Interactive Demo
    # -----------------------------------------------------------------
    with tab_interactive:
        st.subheader("Rate movies in real-time to get instant recommendations")
        st.write("Search for movies you have seen, rate them using the slider, and watch the system build your profile on the fly.")

        col1, col2 = st.columns([2, 1])

        with col1:
            search_query = st.text_input("Search a movie to rate:", placeholder="e.g. Inception, Toy Story, Dark Knight")
            if search_query:
                matches = engine.search_titles(search_query, limit=5)
                if matches:
                    selected_movie = st.selectbox(
                        "Select movie:",
                        matches,
                        format_func=lambda x: x["title"]
                    )
                    user_rating = st.slider("Your rating:", 1.0, 5.0, 4.0, 0.5, key="interactive_slider")
                    
                    if st.button("Add / Update Rating", type="primary"):
                        st.session_state["live_ratings"][selected_movie["movieId"]] = user_rating
                        st.success(f"Saved: {selected_movie['title']} = {user_rating} ⭐")
                else:
                    st.warning("No movie titles matched.")

        with col2:
            st.markdown("#### Your Active Ratings Profile")
            if st.session_state["live_ratings"]:
                rated_list = []
                for mid, r in st.session_state["live_ratings"].items():
                    m_row = engine.movies[engine.movies.movieId == mid]
                    title = m_row.iloc[0]["title"] if not m_row.empty else f"Movie {mid}"
                    rated_list.append({"Title": title, "Rating ⭐": r})
                
                st.dataframe(pd.DataFrame(rated_list), use_container_width=True, hide_index=True)
                if st.button("Clear Ratings"):
                    st.session_state["live_ratings"] = {}
                    st.rerun()
            else:
                st.info("No live ratings added yet. Add a few ratings on the left!")

        st.divider()

        st.subheader("Real-Time Recommendations")
        int_mode = st.radio("Model Method:", ["Hybrid", "Collaborative (SVD)", "Content-Based"], horizontal=True)
        int_top_n = st.slider("Number of results:", 5, 20, 10, key="int_top_n")

        if st.button("Generate Live Recommendations", type="primary"):
            if not st.session_state["live_ratings"]:
                st.warning("Please rate at least one movie first to test live recommendations.")
            else:
                with st.spinner("Projecting taste vector into SVD latent space..."):
                    if int_mode == "Content-Based":
                        results = engine.content_based_from_session(st.session_state["live_ratings"], top_n=int_top_n)
                    elif int_mode == "Collaborative (SVD)":
                        results = engine.collaborative_from_session(st.session_state["live_ratings"], top_n=int_top_n)
                    else:
                        results = engine.hybrid_from_session(st.session_state["live_ratings"], top_n=int_top_n, alpha=0.5)

                if results.empty:
                    st.warning("No recommendations found.")
                else:
                    render_results(results, tmdb_key)

    # -----------------------------------------------------------------
    # TAB 2: Pre-existing MovieLens Users
    # -----------------------------------------------------------------
    with tab_preset:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            user_id = st.selectbox("Select MovieLens User ID:", engine.get_user_ids())
            mode = st.radio("Method:", ["Hybrid", "Content-Based", "Collaborative"], key="preset_mode")
            top_n = st.slider("Number of recommendations:", 5, 20, 10, key="preset_top_n")
            
            title_query = st.text_input("Optional seed movie title:", placeholder="e.g. Dark Knight")
            selected_title = None
            if title_query:
                matches = engine.search_titles(title_query, limit=5)
                if matches:
                    selected_title = st.selectbox("Matching seed title:", [m["title"] for m in matches])

            go = st.button("Fetch Recommendations", type="primary", key="btn_preset")

        with col_b:
            if go:
                with st.spinner("Computing user similarity..."):
                    if mode == "Content-Based":
                        results = (engine.content_based_recommend(selected_title, top_n=top_n)
                                   if selected_title else
                                   engine.content_based_for_user(user_id, top_n=top_n))
                    elif mode == "Collaborative":
                        results = engine.collaborative_recommend(user_id, top_n=top_n)
                    else:
                        results = engine.hybrid_recommend(user_id, title=selected_title, top_n=top_n, alpha=0.5)

                render_results(results, tmdb_key)

    # -----------------------------------------------------------------
    # TAB 3: Browse by Genre
    # -----------------------------------------------------------------
    with tab_browse:
        st.subheader("Top-rated movies by genre (IMDb Weighted Rating)")
        genre = st.selectbox("Genre", ["All"] + engine.get_all_genres())
        browse_n = st.slider("How many to show", 5, 25, 10, key="browse_n")
        browse_results = engine.popular_movies(
            top_n=browse_n, genre=None if genre == "All" else genre
        )
        render_results(browse_results, tmdb_key)


if __name__ == "__main__":
    main()