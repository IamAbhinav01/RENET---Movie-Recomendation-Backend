import pandas as pd
from sqlalchemy import text
from app.config.db_Config import engine
#adddded the movie paht
from app.schemas.postgres_schema import Base

# Ensure tables exist
Base.metadata.create_all(bind=engine)

MOVIES_PATH = "app/dataset/ml-latest-small/movies.csv"
RATINGS_PATH = "app/dataset/ml-latest-small/ratings.csv"


def get_primary_genre(genres_str: str) -> str:
    """Extract the first genre as primary_genre."""
    if not genres_str or genres_str == "(no genres listed)":
        return "Unknown"
    return genres_str.split("|")[0]


def seed():
    print("📂 Loading CSV files...")
    movies_df = pd.read_csv(MOVIES_PATH)
    ratings_df = pd.read_csv(RATINGS_PATH)

    print(f"   → {len(movies_df)} movies loaded")
    print(f"   → {len(ratings_df)} ratings loaded")

    # --- Prepare items ---
    movies_df = movies_df.rename(columns={"movieId": "id"})
    movies_df["primary_genre"] = movies_df["genres"].apply(get_primary_genre)
    movies_df["poster_url"] = None
    movies_df["plot"] = None
    items_df = movies_df[["id", "title", "genres", "primary_genre", "poster_url", "plot"]]

    # --- Prepare users ---
    unique_user_ids = ratings_df["userId"].unique()
    users_df = pd.DataFrame({"id": unique_user_ids})

    # --- Prepare interactions ---
    # Only keep ratings for movies that exist in items
    valid_movie_ids = set(movies_df["id"])
    ratings_df = ratings_df[ratings_df["movieId"].isin(valid_movie_ids)]
    
    interactions_df = ratings_df.rename(columns={
        "userId": "user_id",
        "movieId": "item_id"
    })[["user_id", "item_id", "rating"]]
    interactions_df["event_type"] = "rating"

    with engine.connect() as conn:
        with conn.begin():
            print("\n🗑️  Clearing existing data...")
            conn.execute(text("DELETE FROM interactions"))
            conn.execute(text("DELETE FROM items"))
            conn.execute(text("DELETE FROM users"))

            print("👤 Seeding users...")
            users_df.to_sql("users", conn, if_exists="append", index=False)
            print(f"   → {len(users_df)} users inserted")

            print("🎬 Seeding items (movies)...")
            items_df.to_sql("items", conn, if_exists="append", index=False)
            print(f"   → {len(items_df)} movies inserted")

            print("⭐ Seeding interactions (ratings)...")
            # Insert in batches to avoid memory issues
            batch_size = 5000
            total = len(interactions_df)
            for i in range(0, total, batch_size):
                batch = interactions_df.iloc[i:i + batch_size]
                batch.to_sql("interactions", conn, if_exists="append", index=False)
                print(f"   → Inserted {min(i + batch_size, total)}/{total} ratings...", end="\r")
            print(f"\n   → {total} interactions inserted")

    print("\n✅ Database seeded successfully!")
    print("   You can now run: python main.py")


if __name__ == "__main__":
    seed()
