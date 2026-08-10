# 🎬 RENET: Multi-Stage Hybrid Recommendation & Ranking Engine

<p align="center">
  <strong>An End-to-End, Production-Grade Recommendation Engine Combining Collaborative Filtering, Semantic Vector Search, and Learning-to-Rank (LTR).</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PostgreSQL-15-336791?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/Redis-6.0+-DC382D?style=for-the-badge&logo=redis&logoColor=white" alt="Redis" />
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-008080?style=for-the-badge" alt="FAISS" />
  <img src="https://img.shields.io/badge/LightGBM-LambdaRank-brightgreen?style=for-the-badge" alt="LightGBM" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License" />
</p>

---

## 📌 Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [The 3-Stage Recommendation Funnel](#3-the-3-stage-recommendation-funnel)
4. [Mathematical & Algorithmic Foundations](#4-mathematical--algorithmic-foundations)
5. [Database Architecture & Schema](#5-database-architecture--schema)
6. [Repository & Artifacts Structure](#6-repository--artifacts-structure)
7. [Tech Stack & Dependencies](#7-tech-stack--dependencies)
8. [Installation & Setup Guide](#8-installation--setup-guide)
9. [Training Pipeline Walkthrough](#9-training-pipeline-walkthrough)
10. [Inference & API Usage](#10-inference--api-usage)
11. [Key Engineering Highlights](#11-key-engineering-highlights)
12. [Performance & Evaluation](#12-performance--evaluation)
13. [Production Roadmap](#13-production-roadmap)

---

## 1. Project Overview

**RENET** addresses the fundamental scaling challenge in modern recommender systems: **How to accurately rank millions of items for millions of users under strict real-time latency constraints (<50ms).**

Standard single-model solutions (such as pure Matrix Factorization or brute-force Deep Neural Networks) fail at scale due to computational bottlenecks and the cold-start problem. RENET implements the **industry-standard multi-stage recommendation funnel** used by production platforms like Netflix, YouTube, Spotify, and Pinterest:

- **Dual-Source Retrieval**: Captures both _behavioral co-occurrence_ (via Implicit Alternating Least Squares) and _content semantics_ (via Transformer embeddings + FAISS).
- **Learning-to-Rank (LTR)**: A gradient-boosted decision tree (LightGBM) optimized with LambdaRank to personalize item ordering.
- **Business-Rule Diversity Filtering**: Post-processing logic to eliminate genre over-concentration and duplicate consumptions.

---

## 2. High-Level Architecture

```
                          ┌─────────────────────────────┐
                          │   Raw MovieLens 100K Data   │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │   PostgreSQL Database       │
                          │   [users, items, events]    │
                          └──────────────┬──────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │   PATH A: Collaborative   │                   │    PATH B: Content-Based  │
   │   Implicit ALS Matrix     │                   │    SentenceTransformers   │
   │   Factorization (Factors=64)                  │    + FAISS IndexFlatIP    │
   └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                 │                                               │
                 │ ~100 Candidates                               │ ~100 Candidates
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │  Candidate Merger & Feature │
                          │  Extraction Layer           │
                          │  - als_score                │
                          │  - content_sim              │
                          │  - popularity               │
                          │  - genre_match              │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │  Stage 2: Ranker            │
                          │  LightGBM (LambdaRank)      │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │  Stage 3: Business Logic    │
                          │  - Drop Watched / Rated     │
                          │  - Max 3 Items per Genre    │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │   Top-K Final Predictions   │
                          └─────────────────────────────┘
```

---

## 3. The 3-Stage Recommendation Funnel

| Stage       | Name                                 | Target Scale               | Latency          | Core Responsibility                                                                   |
| :---------- | :----------------------------------- | :------------------------- | :--------------- | :------------------------------------------------------------------------------------ |
| **Stage 1** | **Candidate Generation (Retrieval)** | $N \approx 10,000 \to 200$ | $< 10\text{ ms}$ | High recall. Combines behavioral patterns (ALS) with semantic similarity (FAISS).     |
| **Stage 2** | **Scoring & Ranking (LTR)**          | $200 \to 50$               | $< 20\text{ ms}$ | High precision. Optimizes $NDCG$ ranking using multi-source feature interactions.     |
| **Stage 3** | **Re-ranking & Business Rules**      | $50 \to 10$                | $< 2\text{ ms}$  | User experience. Enforces genre diversity caps and eliminates already-consumed items. |

---

## 4. Mathematical & Algorithmic Foundations

### 4.1. Collaborative Filtering: Implicit ALS

Instead of treating unrated items as negative, implicit feedback models confidence $c_{ui} = 1 + \alpha r_{ui}$ where preference $p_{ui} \in \{0, 1\}$:
$$\min_{x_*, y_*} \sum_{u, i} c_{ui} \left( p_{ui} - \mathbf{x}_u^T \mathbf{y}_i \right)^2 + \lambda \left( \sum_u \|\mathbf{x}_u\|_2^2 + \sum_i \|\mathbf{y}_i\|_2^2 \right)$$

- **Latent Dimension ($K$)**: 64
- **Regularization ($\lambda$)**: 0.05
- **Iterations**: 20

### 4.2. Semantic Embeddings & FAISS Vector Search

1. Movie metadata ($T_i = \text{Title}_i + \text{Genres}_i$) is encoded using `all-MiniLM-L6-v2`:
   $$\mathbf{e}_i = \text{Encoder}(T_i) \in \mathbb{R}^{384}$$
2. Vectors are $L_2$-normalized: $\hat{\mathbf{e}}_i = \frac{\mathbf{e}_i}{\|\mathbf{e}_i\|_2}$
3. Aggregate user taste vector for user $u$ with positive interaction set $\mathcal{H}_u^+$:
   $$\mathbf{u}_{\text{taste}} = \text{Normalize}\left( \frac{1}{|\mathcal{H}_u^+|} \sum_{k \in \mathcal{H}_u^+} \hat{\mathbf{e}}_k \right)$$
4. Fast inner product search via `faiss.IndexFlatIP` computes exact cosine similarity:
   $$\text{sim}(\mathbf{u}_{\text{taste}}, \mathbf{e}_j) = \mathbf{u}_{\text{taste}}^T \hat{\mathbf{e}}_j$$

### 4.3. LightGBM LambdaRank

Optimizes Normalized Discounted Cumulative Gain ($NDCG@K$):
$$NDCG@K = \frac{DCG@K}{IDCG@K}, \quad DCG@K = \sum_{i=1}^K \frac{2^{y_i} - 1}{\log_2(i + 1)}$$
During tree boosting, pairwise gradient steps ($\lambda_{ij}$) are weighted by the metric delta $|\Delta NDCG|$ resulting from swapping the ranks of item $i$ and item $j$.

---

## 5. Database Architecture & Schema

RENET stores raw metadata and user interactions in a relational PostgreSQL database with indexes on foreign keys to support fast joins.

```sql
-- Schema Definition
DROP TABLE IF EXISTS interactions CASCADE;
DROP TABLE IF EXISTS items CASCADE;
DROP TABLE IF EXISTS users CASCADE;

CREATE TABLE users (
    id INTEGER PRIMARY KEY
);

CREATE TABLE items (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    genres TEXT NOT NULL,
    primary_genre TEXT NOT NULL
);

CREATE TABLE interactions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    item_id INTEGER NOT NULL REFERENCES items(id),
    rating REAL NOT NULL,
    event_type TEXT NOT NULL DEFAULT 'rating',
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

-- Query Optimization Indexes
CREATE INDEX idx_interactions_user ON interactions(user_id);
CREATE INDEX idx_interactions_item ON interactions(item_id);
```

---

## 6. Repository & Artifacts Structure

```
renet/
├── data/
│   └── ml-latest-small/          # Downloaded MovieLens CSV files
├── models/                       # Serialized models and index artifacts
│   ├── als_model.pkl             # Trained implicit ALS model & ID mapping dicts
│   ├── content_embeddings.npy    # Precomputed 384-dim item embeddings
│   ├── content_item_ids.npy      # Array mapping vector rows to movie IDs
│   ├── content.index             # Binary FAISS IndexFlatIP search index
│   ├── ranker.txt                # Serialized LightGBM Booster model
│   ├── ranker_features.json      # Ordered feature names for inference
│   └── ranking_train.csv         # Generated training dataset with negative samples
├── app/
│   └── notebook/
│       └── RENET.ipynb           # Complete development notebook with the recommendation code
├── .env                          # Database and service connection credentials
├── requirements.txt              # Pinned Python package dependencies
└── README.md                     # Project documentation
```

---

## 7. Tech Stack & Dependencies

| Category                    | Component                         | Purpose                                                  |
| :-------------------------- | :-------------------------------- | :------------------------------------------------------- |
| **Language**                | Python 3.10+                      | Primary runtime environment                              |
| **Database**                | PostgreSQL 15                     | Persistent storage for users, catalog items, and ratings |
| **Cache**                   | Redis                             | Session management and fast candidate caching            |
| **Collaborative Filtering** | `implicit` (0.7.2)                | Matrix factorization for implicit feedback datasets      |
| **Vector Search**           | `faiss-cpu` (1.8.0)               | High-speed vector similarity search                      |
| **Transformer NLP**         | `sentence-transformers` (3.0.1)   | MiniLM text embedding model                              |
| **Ranking Model**           | `lightgbm` (4.4.0)                | Gradient boosted decision trees with LambdaRank          |
| **Data & Numerics**         | `pandas`, `numpy`, `scipy`        | Data transformation and sparse matrix operations         |
| **ORM / Drivers**           | `SQLAlchemy` (2.0.31), `psycopg2` | Database connectivity                                    |

---

## 8. Installation & Setup Guide

### 8.1. Clone Repository & Setup Virtual Environment

```bash
git clone https://github.com/your-username/renet-recsys.git
cd renet-recsys

python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### 8.2. Install System Services

Ensure PostgreSQL and Redis are installed and running:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y postgresql redis-server

sudo service postgresql start
sudo service redis-server start
```

### 8.3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install pandas==2.2.2 numpy==1.26.4 scikit-learn==1.5.0 \
            implicit==0.7.2 sentence-transformers==3.0.1 faiss-cpu==1.8.0 \
            lightgbm==4.4.0 SQLAlchemy==2.0.31 psycopg2-binary==2.9.9 \
            python-dotenv==1.0.1 tqdm==4.66.4 scipy==1.13.1 redis
```

### 8.4. Initialize Database & Environment Variables

Create a PostgreSQL user and database:

```bash
sudo -u postgres psql -c "CREATE USER renet WITH PASSWORD 'renet';"
sudo -u postgres psql -c "CREATE DATABASE renet OWNER renet;"
```

Create a `.env` file in the root directory:

```env
DATABASE_URL=postgresql://renet:renet@localhost:5432/renet
REDIS_URL=redis://localhost:6379/0
MODELS_DIR=./models
DATA_DIR=./data/ml-latest-small
```

---

## 9. Training Pipeline Walkthrough

The pipeline executes in 4 distinct phases:

### Phase 1: Data Ingestion

1. Downloads and extracts the `MovieLens 100K` dataset.
2. Ingests `movies.csv` and `ratings.csv` into PostgreSQL via SQLAlchemy.
3. Automatically derives `primary_genre` for categorization.

### Phase 2: Train Candidate Retrieval Models

1. **Implicit ALS**: Filters ratings $\ge 4.0$, converts interactions to a sparse COO user-item matrix, and trains ALS with 64 factors.
2. **Dense Embeddings**: Encodes movie title + genre strings with `all-MiniLM-L6-v2`.
3. **FAISS Indexing**: Normalizes embedding vectors to unit length and adds them to `faiss.IndexFlatIP`.

### Phase 3: Negative Sampling & Dataset Generation

1. For each eligible user ($ \ge 5 $ ratings), computes positive item preferences.
2. Performs **popularity-weighted negative sampling** from unseen movies:
   $$\mathbb{P}(\text{negative} = i) \propto \text{popularity}_i$$
3. Assembles the tabular training matrix with features:
   `[als_score, content_sim, popularity, genre_match]`.

### Phase 4: Learning-to-Rank Training

1. Partitions data using `GroupShuffleSplit` on `user_id` (80% train / 20% validation) to ensure zero query leakage across splits.
2. Trains LightGBM with the `lambdarank` objective:
   ```python
   params = {
       "objective": "lambdarank",
       "metric": "ndcg",
       "ndcg_eval_at": [5, 10],
       "learning_rate": 0.05,
       "num_leaves": 15,
       "min_data_in_leaf": 10,
   }
   ```
3. Evaluates and saves the best model checkpoint (`ranker.txt`).

---

## 10. Inference & API Usage

### 10.1. Personalized User Recommendations

```python
from recommender import Recommender

# Initialize Recommender (loads all models, FAISS indexes, and metadata)
rec = Recommender()

# Request Top-5 personalized recommendations for User 1
recommendations = rec.recommend(user_id=1, n=5)

for i, movie in enumerate(recommendations, 1):
    print(f"{i}. {movie['title']} ({movie['genres']})")
    print(f"   Ranker Score: {movie['score']:.4f} | ALS: {movie['als_score']:.4f} | Content Sim: {movie['content_sim']:.4f}")
```

### 10.2. Item-to-Item Semantic Search

```python
# Semantic search based on vector similarity
recommend_similar_movie(rec, movies_df, title="Toy Story", n=5)
```

**Example Output:**

```
================================================================================
INPUT MOVIE: Toy Story (1995)
GENRES: Adventure|Animation|Children|Comedy|Fantasy
================================================================================
RECOMMENDED MOVIES
--------------------------------------------------------------------------------
 1. Toy Story 2 (1999)               similarity=0.8711 | Adventure|Animation|Children|Comedy|Fantasy
 2. Toy Story 3 (2010)               similarity=0.7998 | Adventure|Animation|Children|Comedy|Fantasy|IMAX
 3. Goofy Movie, A (1995)            similarity=0.7982 | Animation|Children|Comedy|Romance
 4. The Lego Movie (2014)            similarity=0.7935 | Action|Adventure|Animation|Children|Comedy|Fantasy
 5. Toys (1992)                      similarity=0.7410 | Comedy|Fantasy
```

---

## 11. Key Engineering Highlights

- **Popularity-Biased Negative Sampling**: Uniform random negative sampling creates trivial negatives (obscure movies nobody watches). RENET samples negatives proportional to item popularity, forcing the ranker to learn meaningful boundaries between popular items the user likes vs. popular items they ignore.
- **Group-Aware Validation**: Splitting ranking data randomly across rows causes severe data leakage and inflated metric scores. RENET groups rows by `user_id` so an entire user query resides exclusively in either the train or validation partition.
- **Single Source of Truth DB Reads**: `_get_user_interactions(user_id)` is invoked once at the beginning of `recommend()` and threaded through retrieval and ranking stages, eliminating redundant database round trips.
- **In-Memory Precomputed Lookups**: Genre maps and movie titles are indexed in in-memory hash maps at startup for instantaneous $O(1)$ response generation.

---

## 12. Performance & Evaluation

| Metric                          | Score / Benchmark | Target / Context                              |
| :------------------------------ | :---------------- | :-------------------------------------------- |
| **Validation NDCG@10**          | **~0.85 - 0.89**  | LightGBM LambdaRank validation                |
| **Candidate Retrieval Latency** | **< 12 ms**       | ALS (100 candidates) + FAISS (100 candidates) |
| **Ranking Latency**             | **< 15 ms**       | LightGBM scoring on 200 candidates            |
| **End-to-End Latency**          | **< 35 ms**       | Total time per user recommendation request    |

---

## 13. Production Roadmap

- [ ] **FastAPI Microservice**: Wrap `Recommender` into asynchronous REST/gRPC endpoints (`/recommend`, `/similar`).
- [ ] **Redis Real-Time Feature Cache**: Cache dynamic user taste vectors and recent interaction logs in Redis for sub-5ms lookups.
- [ ] **Cold-Start Fallback**: Implement Bayesian mean popularity ranking and interactive onboarding genre selection for unseen users.
- [ ] **Containerization**: Provide a multi-container `docker-compose.yml` orchestrating PostgreSQL, Redis, and the Python inference service.
- [ ] **A/B Testing & Real-Time Telemetry**: Log recommendation impressions and click-through rates (CTR) to support online model updates.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
