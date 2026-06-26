import os
import joblib
import pandas as pd
import mlflow
import mlflow.xgboost  # <-- Changed this to native xgboost tracking
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Relative paths to locate your clean data and save your model
MODEL_DIR = os.path.join("model", "flight_prediction")
DATA_PATH = os.path.join("data", "flight_prediction", "processed_flights.csv")

print("🏋️‍♂️ Loading processed data and starting model training...")

# Load the clean data you generated in Step 2
df = pd.read_csv(DATA_PATH)

# X = Features (distance, month, day, etc.), y = Target (price)
X = df.drop("price", axis=1)
y = df["price"]

# Save the exact order of columns so our Streamlit web app doesn't mix them up
joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "feature_columns.pkl"))

# Split into 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start an MLflow experiment tracking session
with mlflow.start_run():

    # Hyperparameters (settings) for the XGBoost brain
    params = {
        'n_estimators': 500,
        'learning_rate': 0.08,
        'max_depth': 5,
        'subsample': 0.88,
        'random_state': 42
    }

    # Log the settings to our MLflow database
    mlflow.log_params(params)

    # Initialize and train the XGBoost Model
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Save a physical backup copy of the trained model file
    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_regressor.pkl"))

    # Quiz the model on the remaining 20% test data
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log the test scores to our MLflow database
    mlflow.log_metric('mse', mse)
    mlflow.log_metric('r2', r2)

    # Register the model using the native XGBoost framework to bypass the security wall
    mlflow.xgboost.log_model(model, artifact_path='xgb_model')

    print("=========================================")
    print(f"✅ Training Complete!")
    print(f"📊 Model Performance Quiz Results:")
    print(f"   - Mean Squared Error (MSE): {mse:.2f}")
    print(f"   - R-squared Accuracy (R2): {r2:.4f} ({r2*100:.1f}% accurate)")
    print("=========================================")