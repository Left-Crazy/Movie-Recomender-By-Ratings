# 🎬 Hybrid Movie Recommendation Engine

A high-performance Movie Recommendation Engine built with **Python**, **scikit-learn**, and **Streamlit**. The system combines **Content-Based Filtering (TF-IDF)** and **Collaborative Filtering (TruncatedSVD Matrix Factorization)** to deliver personalized movie suggestions. 

It features an interactive **Real-Time SVD Fold-In Projection** mechanism, allowing live users to rate movies in the UI and receive updated recommendations instantly without retraining the base model.

---

## 🌐 Live Web Demo

Access and test the live application directly in your web browser without installing dependencies locally:

👉 **[Launch Live Web Application]<https://movie-recomender-by-ratings-4vbtwhcgwwqqdq5xtan9zr.streamlit.app/>>**  
*(Note: Replace the link above with your deployed Streamlit Community Cloud URL)*

---

## 🌟 Key Features

* **⚡ Real-Time SVD Fold-In Projection:** Dynamically projects user ratings entered via UI sliders into latent vector space for instant recommendations without full model retraining.
* **👤 Pre-Existing User Profiles:** Predicts ratings and preferences for historical MovieLens users using SVD matrix factorization.
* **🎯 Seed Movie Lookup:** Generates high-accuracy similarity matches for any typed movie title.
* **⚖️ Custom Hybrid Weighting:** Blends content similarity vectors and collaborative rating predictions via an adjustable weighting factor ($\alpha$).
* **🖼️ Live TMDB Poster Integration:** Fetches official movie artwork using The Movie Database (TMDB) API.
* **🔥 Cold-Start Handling:** Uses IMDb-style weighted average popularity fallback for unrated items or new users.

---

## 🛠️ Architecture & Mathematical Foundation
┌─────────────────┐     ┌───────────────────────┐     ┌──────────────────────┐
│ Content Engine  │ ──> │  TF-IDF Vectorizer    │ ──> │ Cosine Similarity    │
│ (Genres/Titles) │     │  (Sparse Text Matrix) │     │ Vector Space         │
└─────────────────┘     └───────────────────────┘     └──────────────────────┘
│
▼
┌─────────────────┐     ┌───────────────────────┐     ┌──────────────────────┐
│  Interactive    │ ──> │ Real-Time SVD Fold-In │ ──> │ Hybrid Scoring Engine│
│   Streamlit UI  │     │ Latent Space q=(r-μ)V │     │  α·S_cont + (1-α)S_cf│
└─────────────────┘     └───────────────────────┘     └──────────────────────┘

### 1. Content-Based Filtering (TF-IDF)
Movie titles and genre strings are tokenized into a term frequency-inverse document frequency matrix. Similarity between items is computed via Cosine Similarity:

$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\Vert{}\mathbf{u}\Vert{} \Vert{}\mathbf{v}\Vert{}}$$

### 2. Collaborative Filtering (Truncated SVD Fold-In)
The historical user-item rating matrix $\mathbf{R}_{\text{centered}}$ is decomposed into latent matrices:

$$\mathbf{R}_{\text{centered}} \approx \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

For a live user session with rating vector $\mathbf{r}$, the taste profile $\mathbf{q}$ is projected into the latent space on the fly:

$$\mathbf{q} = (\mathbf{r} - \mu) \mathbf{V}$$

Predicted ratings are generated via reconstruction: $\mathbf{\hat{r}} = \mathbf{q} \mathbf{V}^T + \mu$.

### 3. Hybrid Scoring
The final recommendation score balances content relevance and collaborative feedback:

$$\text{Score}_{\text{hybrid}} = \alpha \cdot S_{\text{content}} + (1 - \alpha) \cdot S_{\text{collaborative}}$$

---

## 💻 Local Setup & Execution Guide

Follow these steps to run the application locally on your machine.

### 1. Prerequisites
* Python 3.9 or higher installed.

### 2. Clone the Repository
```bash
git clone [https://github.com/your-username/movie-recommendation-engine.git](https://github.com/your-username/movie-recommendation-engine.git)
cd movie-recommendation-engine

3. Install Dependencies
Bash
pip install -r requirements.txt
(If requirements.txt is missing, install manually: pip install pandas numpy scipy scikit-learn streamlit requests)

4. Fetch MovieLens Dataset
Run the automated downloader script to retrieve and format the official MovieLens dataset:

Bash
python download_real_data.py

5. Launch the Streamlit Dashboard
Bash
python -m streamlit run app.py
Open your browser and navigate to http://localhost:8501.