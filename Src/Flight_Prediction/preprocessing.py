import os
import pandas as pd
import joblib
from sklearn.preprocessing import TargetEncoder

# =====================
# CONFIG (Relative Paths)
# =====================
DATA_DIR = os.path.join("data", "flight_prediction")
MODEL_DIR = os.path.join("model", "flight_prediction")

os.makedirs(MODEL_DIR, exist_ok=True)

FLIGHT_TYPE_MAP = {
    "economy": 0,
    "business": 1,
    "firstClass": 2
}

print("🚀 Starting preprocessing...")

# =====================
# LOAD RAW DATA
# =====================
df = pd.read_csv(os.path.join(DATA_DIR, "flights.csv"))

# Clean data types
df['price'] = df['price'].astype(int)
df['date'] = pd.to_datetime(df['date'])

# Drop raw tracking columns we don't need for math
df.drop(columns=['travelCode', 'userCode'], inplace=True, errors='ignore')

# Break down the date into separate Day, Month, and Year columns
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# ==========================================
# ✨ NEW: UI DROPDOWN EXTRACTOR (DYNAMIC)
# ==========================================
# Extract every single unique city, agency, and year BEFORE we encode them into numbers
unique_cities = sorted(list(set(df["from"].unique().tolist() + df["to"].unique().tolist())))
unique_agencies = sorted(df["agency"].unique().tolist())
unique_years = sorted(df["year"].unique().tolist())

ui_dropdown_options = {
    "cities": unique_cities,
    "agencies": unique_agencies,
    "years": unique_years
}
# Save this dictionary so our Streamlit app can read it instantly
joblib.dump(ui_dropdown_options, os.path.join(MODEL_DIR, "ui_dropdown_options.pkl"))

# Drop original date column now that we extracted features
df.drop('date', axis=1, inplace=True)

# =====================
# TARGET ENCODERS
# =====================
from_encoder = TargetEncoder(target_type="continuous")
to_encoder = TargetEncoder(target_type="continuous")
agency_encoder = TargetEncoder(target_type="continuous")

df["from"] = from_encoder.fit_transform(df[["from"]], df["price"])
df["to"] = to_encoder.fit_transform(df[["to"]], df["price"])
df["agency"] = agency_encoder.fit_transform(df[["agency"]], df["price"])

df["flightType"] = df["flightType"].map(FLIGHT_TYPE_MAP)

# =====================
# SAVE ARTIFACTS
# =====================
joblib.dump(from_encoder, os.path.join(MODEL_DIR, "from_encoder.pkl"))
joblib.dump(to_encoder, os.path.join(MODEL_DIR, "to_encoder.pkl"))
joblib.dump(agency_encoder, os.path.join(MODEL_DIR, "agency_encoder.pkl"))
joblib.dump(FLIGHT_TYPE_MAP, os.path.join(MODEL_DIR, "flight_type_map.pkl"))

# Save the final cleaned dataset
df.to_csv(os.path.join(DATA_DIR, "processed_flights.csv"), index=False)

print("✅ Preprocessing completed successfully! Dynamic dropdown lists saved.")