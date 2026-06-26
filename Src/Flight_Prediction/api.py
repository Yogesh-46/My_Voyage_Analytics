import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Model Artifact Registries
REGRESSION_DIR = os.path.join("Model", "flight_prediction")
CLASSIFICATION_DIR = os.path.join("Model", "flight_classification")
RECOMMENDER_DIR = os.path.join("Model", "hotel_recommendation")

print("🔌 Booting Multi-Model MLOps REST API Gateway...")

try:
    # 1. Load Regression Engine
    reg_model = joblib.load(os.path.join(REGRESSION_DIR, "xgb_regressor.pkl"))
    from_encoder = joblib.load(os.path.join(REGRESSION_DIR, "from_encoder.pkl"))
    to_encoder = joblib.load(os.path.join(REGRESSION_DIR, "to_encoder.pkl"))
    agency_encoder = joblib.load(os.path.join(REGRESSION_DIR, "agency_encoder.pkl"))
    flight_type_map = joblib.load(os.path.join(REGRESSION_DIR, "flight_type_map.pkl"))
    feature_columns = joblib.load(os.path.join(REGRESSION_DIR, "feature_columns.pkl"))
    
    # 2. Load Classification Engine
    clf_model = joblib.load(os.path.join(CLASSIFICATION_DIR, "rf_classifier.pkl"))
    company_encoder = joblib.load(os.path.join(CLASSIFICATION_DIR, "company_encoder.pkl"))
    
    # 3. Load Recommender Artifacts
    rec_artifacts = joblib.load(os.path.join(RECOMMENDER_DIR, "recommender_artifacts.pkl"))
    item_similarity_df = rec_artifacts["similarity_matrix"]
    user_item_matrix = rec_artifacts["user_item_matrix"]
    hotel_place_map = rec_artifacts["raw_hotels_data"]
    
    print("✅ All 3 ML Models initialized and loaded successfully. Gateway Live.")
    print(f"📊 Expected Regression Columns: {list(feature_columns)}")
except FileNotFoundError as e:
    print(f"❌ Critical Error: Missing serialized artifacts! {e}")

@app.route("/predict", methods=["POST"])
def predict_price():
    try:
        json_data = request.get_json()
        
        # Build base frame from the incoming request payload
        df_input = pd.DataFrame([json_data])
        
        # 🛡️ Defensive Transformation Loop: Safely tries 2D transformation, falls back to 1D if needed
        for col, encoder in [("from", from_encoder), ("to", to_encoder), ("agency", agency_encoder)]:
            if col in df_input.columns:
                try:
                    df_input[col] = encoder.transform(df_input[[col]]).ravel()
                except Exception:
                    df_input[col] = encoder.transform(df_input[col])
        
        # Map flight class string tokens to numeric elements
        if "flightType" in df_input.columns:
            df_input["flightType"] = df_input["flightType"].map(flight_type_map)
            
        # 🛡️ Missing Feature Guard: Fills columns like distance/duration with 0 if missing from frontend
        for expected_col in feature_columns:
            if expected_col not in df_input.columns:
                df_input[expected_col] = 0
                
        # Slice and reorder columns to match original model training footprint exactly
        df_final = df_input[feature_columns]
        
        prediction = reg_model.predict(df_final)[0]
        return jsonify({"status": "success", "predicted_price": float(prediction)}), 200
    except Exception as e:
        import traceback
        print("\n🚨 --- UNINTERRUPTED CRITICAL BACKEND TRACEBACK --- 🚨")
        traceback.print_exc()
        print("🚨 -------------------------------------------- 🚨\n")
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/classify_user", methods=["POST"])
def classify_user():
    try:
        json_data = request.get_json()
        df_input = pd.DataFrame([json_data])
        df_input['company_encoded'] = company_encoder.transform(df_input['company'])
        features = df_input[['age', 'company_encoded']]
        prediction = clf_model.predict(features)[0]
        gender_output = "Male" if prediction == 0 else "Female"
        return jsonify({"status": "success", "predicted_gender": gender_output}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/recommend_hotels", methods=["POST"])
def recommend_hotels():
    try:
        json_data = request.get_json()
        user_code = int(json_data.get("userCode", 0))
        dest_city = json_data.get("destination_city")
        
        if user_code not in user_item_matrix.index:
            city_hotels = [hotel for hotel, city in hotel_place_map.items() if city == dest_city]
            return jsonify({"status": "success", "recommendations": city_hotels[:3], "type": "Popularity Fallback"}), 200
            
        user_vector = user_item_matrix.loc[user_code].copy()
        predicted_scores = user_vector.values @ item_similarity_df.values
        scores_series = pd.Series(predicted_scores, index=user_item_matrix.columns)
        
        sorted_hotels = scores_series.sort_values(ascending=False).index.tolist()
        filtered_recommendations = [hotel for hotel in sorted_hotels if hotel_place_map.get(hotel) == dest_city]
        
        return jsonify({"status": "success", "recommendations": filtered_recommendations[:3], "type": "Collaborative Filtering"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    # 🎯 FIXED: use_reloader=False stops Watchdog from killing log outputs during error states
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)