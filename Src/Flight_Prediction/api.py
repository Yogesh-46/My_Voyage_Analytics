import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask application
app = Flask(__name__)

# Relative paths to locate your saved AI artifacts
MODEL_DIR = os.path.join("model", "flight_prediction")

print("🔌 Loading AI model components into Flask backend server...")

# Load models and encoders once when the server boots up
try:
    model = joblib.load(os.path.join(MODEL_DIR, "xgb_regressor.pkl"))
    from_encoder = joblib.load(os.path.join(MODEL_DIR, "from_encoder.pkl"))
    to_encoder = joblib.load(os.path.join(MODEL_DIR, "to_encoder.pkl"))
    agency_encoder = joblib.load(os.path.join(MODEL_DIR, "agency_encoder.pkl"))
    flight_type_map = joblib.load(os.path.join(MODEL_DIR, "flight_type_map.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    print("✅ All AI artifacts loaded successfully. API is ready.")
except FileNotFoundError:
    print("❌ Error: Missing model artifacts! Run preprocessing.py and train.py first.")

# Define a Health Check Route (Great for Kubernetes later!)
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model": "XGBoost Flight Regressor v1.0"}), 200

# Define the Core Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Grab the raw JSON data sent over the network
        json_data = request.get_json()
        
        # 2. Convert incoming JSON dictionary into a Pandas DataFrame row
        df_input = pd.DataFrame([json_data])
        
        # 3. Process categorical strings using your fitted target encoders
        df_input["from"] = from_encoder.transform(df_input[["from"]])
        df_input["to"] = to_encoder.transform(df_input[["to"]])
        df_input["agency"] = agency_encoder.transform(df_input[["agency"]])
        df_input["flightType"] = df_input["flightType"].map(flight_type_map)
        
        # 4. Enforce strict column order matching the XGBoost training pattern
        df_input = df_input[feature_columns]
        
        # 5. Compute the numerical prediction
        prediction = model.predict(df_input)[0]
        
        # 6. Return the answer as a clean JSON package
        return jsonify({
            "status": "success",
            "predicted_price": float(prediction)
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# Fire up the network server
if __name__ == "__main__":
    # We run on port 5001 to avoid any conflicts with MLflow or Streamlit ports
    app.run(host="0.0.0.0", port=5001, debug=True)