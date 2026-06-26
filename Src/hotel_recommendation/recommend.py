import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configuration paths matching your exact repository casing
DATA_FILE = os.path.join("Data", "Hotels", "hotels.csv")
MODEL_DIR = os.path.join("Model", "hotel_recommendation")
os.makedirs(MODEL_DIR, exist_ok=True)

print("🛰️ Ingesting hotels.csv for collaborative filtering matrix construction...")
df_hotels = pd.read_csv(DATA_FILE)

# Step 1: Create a pivot matrix tracking how many times each user stayed at each hotel
print("🔄 Constructing User-Item Interaction frequency matrix...")
user_item_matrix = df_hotels.groupby(['userCode', 'name']).size().unstack(fill_value=0)

# Step 2: Compute Item-to-Item Cosine Similarity based on user co-occurrence patterns
print("🧮 Calculating Item-Item Cosine Similarity across all properties...")
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(
    item_similarity, 
    index=user_item_matrix.columns, 
    columns=user_item_matrix.columns
)

# ==========================================
# 📊 RECOMMENDER EVALUATION ENGINE (Hit Rate @ 3)
# ==========================================
print("🧪 Evaluating Recommender performance via cross-validation split...")
hits = 0
total = 0

# Test evaluation loop across a sample vector of users
for user_id in user_item_matrix.index[:300]:
    user_vector = user_item_matrix.loc[user_id].copy()
    interacted_items = user_vector[user_vector > 0].index.tolist()
    
    # We need at least 2 interactions to hide one for evaluation
    if len(interacted_items) < 2:
        continue
        
    # Hide one item as the ground-truth target
    test_item = np.random.choice(interacted_items)
    user_vector[test_item] = 0 
    
    # Compute score profiles: Dot product of user history with item similarity patterns
    predicted_scores = user_vector.values @ item_similarity_df.values
    scores_series = pd.Series(predicted_scores, index=user_item_matrix.columns)
    
    # Exclude items the user has already stayed at during training stage
    scores_series = scores_series.drop(index=user_vector[user_vector > 0].index)
    
    # Extract the top 3 recommendations
    top_3_recommendations = scores_series.nlargest(3).index.tolist()
    
    if test_item in top_3_recommendations:
        hits += 1
    total += 1

hit_rate = (hits / total) * 100 if total > 0 else 0.0

print("\n=========================================")
print(f"✅ Collaborative Recommender Engineered!")
print(f"🎯 Hit Rate @ 3 Validation Score: {hit_rate:.2f}%")
print("=========================================")

# Save the similarity matrices and interaction histories for the app interface
artifacts = {
    "similarity_matrix": item_similarity_df,
    "user_item_matrix": user_item_matrix,
    "raw_hotels_data": df_hotels[['name', 'place']].drop_duplicates().set_index('name')['place'].to_dict()
}
joblib.dump(artifacts, os.path.join(MODEL_DIR, "recommender_artifacts.pkl"))
print("💾 Collaborative filtering artifacts saved successfully in Model/ directory.")