# ReNet Recommendation Service

A FastAPI-based movie recommendation microservice that combines collaborative filtering, content similarity, and ranking logic to generate personalized recommendations.

## Overview

ReNet is a hybrid recommendation system inspired by modern recommendation architectures used by streaming platforms. It blends:

- collaborative filtering via Implicit ALS
- semantic candidate retrieval via FAISS vector search
- learning-to-rank scoring via LightGBM
- business rules such as filtering already watched items and limiting repeated genres

The application exposes a lightweight REST API and is built to work with PostgreSQL for persistent data and Redis for optional caching.

## Architecture

![ReNet Architecture](architecture.png)

The design follows this flow:

1. A client sends a recommendation request to the FastAPI router.
2. The app checks Redis for a cached result.
3. If no cache hit exists, the ML pipeline runs.
4. The pipeline generates candidates using ALS and FAISS.
5. The ranker scores candidates and applies business rules.
6. The final list is returned to the user and optionally cached.

## System components

- API layer: FastAPI router and HTTP endpoints
- Service layer: recommendation and operation logic
- Repository layer: database access for users, items, and interactions
- ML pipeline: ALS, FAISS, and LightGBM scoring
- Data sources: PostgreSQL for catalog and interaction data, Redis for cache
- Artifacts: serialized model files and indexes stored under the app artifacts folder

## Project structure

```text
ReNet_Recommendation/
├── app/
│   ├── artifacts/
│   │   ├── als_model.pkl
│   │   ├── content.index
│   │   ├── content_embeddings.npy
│   │   ├── content_item_ids.npy
│   │   ├── ranker.txt
│   │   ├── ranker_features.json
│   │   └── ranking_train.csv
│   ├── config/
│   │   ├── artifacts_loader.py
│   │   ├── db_Config.py
│   │   ├── logger_Config.py
│   │   ├── reddis_config.py
│   │   └── server_Config.py
│   ├── repository/
│   │   ├── crudOperations.py
│   │   ├── interactionRepository.py
│   │   ├── itemsRepository.py
│   │   └── userRepository.py
│   ├── router/
│   │   └── operations.py
│   ├── schemas/
│   │   └── postgres_schema.py
│   ├── services/
│   │   ├── operationService.py
│   │   ├── recommendations.py
│   │   └── retrainService.py
│   └── dataset/
│       └── ml-latest-small/
├── main.py
├── seed_db.py
├── pyproject.toml
├── requirements.txt
├── .env
├── architecture.png
├── architecture.md
├── build.md
├── detailed.md
├── LICENSE
├── README.md
└── tests/
    └── test_operation_service.py
```

## Tech stack

- Python 3.10+
- FastAPI
- SQLAlchemy
- PostgreSQL
- Redis
- NumPy / Pandas
- FAISS
- implicit ALS
- LightGBM

## Recommendation flow

The recommendation logic in [app/services/recommendations.py](app/services/recommendations.py) follows three stages:

### 1. Candidate generation

- ALS model generates collaborative filtering candidates
- FAISS index finds semantically related content candidates
- both sets are merged into a broader candidate pool

### 2. Ranking

- feature values such as ALS score, content similarity, popularity, and genre match are extracted
- LightGBM ranks the merged candidate list

### 3. Business rules

- previously watched movies are filtered out
- no more than three movies per genre are kept
- final result is truncated to the requested top K

## API endpoints

The application currently exposes these endpoints from [app/router/operations.py](app/router/operations.py):

```http
GET /
POST /admin/reload-models
GET /interaction
GET /item
GET /api/recommend?user_id=1&n=3
```

### Example request

```bash
curl "http://localhost:8000/api/recommend?user_id=1&n=3"
```

### Example response

```json
{
  "user_id": 1,
  "recommendations": [
    {
      "id": 1148,
      "title": "Wallace & Gromit: The Wrong Trousers (1993)",
      "genres": "Animation|Children|Comedy|Crime",
      "score": 0.09605650564711671
    }
  ]
}
```

## Local setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd ReNet_Recommendation
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

You can also use the project environment defined in the repository configuration:

```bash
uv sync
```

### 4. Configure environment variables

Create or update a `.env` file with your database URL:

```env
DB_URL=postgresql://your_user:your_password@localhost/renet
```

Optional Redis settings are handled in the config layer and can be used if Redis is available.

### 5. Start the API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000/docs
```

## Data model

The system relies on PostgreSQL tables such as:

- users
- items
- interactions

The schema is modeled in [app/schemas/postgres_schema.py](app/schemas/postgres_schema.py).

## Notes

- The recommendation service loads model artifacts on startup.
- Redis is optional for local usage and acts as a cache layer when available.
- The service is designed to be extended with additional endpoints, retraining jobs, and a frontend client.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
sudo -u postgres psql -c "CREATE DATABASE renet OWNER renet;"

````

Create a `.env` file in the root directory:

```env
DATABASE_URL=postgresql://renet:renet@localhost:5432/renet
REDIS_URL=redis://localhost:6379/0
MODELS_DIR=./models
DATA_DIR=./data/ml-latest-small
````

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
