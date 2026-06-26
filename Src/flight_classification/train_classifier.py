import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 📁 Configure paths to point directly to your new data subfolder
# CHANGE "users" to your exact folder name if it is different!
DATA_FILE = os.path.join("Data", "users", "users.csv") 
MODEL_DIR = os.path.join("Model", "flight_classification")
os.makedirs(MODEL_DIR, exist_ok=True)

print("🛰️ Ingesting real users.csv for demographic classification training...")
df = pd.read_csv(DATA_FILE)

# Clean data: Filter out unlabelled 'none' entries to train a clean binary classifier
df_clean = df[df['gender'].isin(['male', 'female'])].copy()

# Feature Engineering: Encode the 'company' categorical text into numbers
company_encoder = LabelEncoder()
df_clean['company_encoded'] = company_encoder.fit_transform(df_clean['company'])

# Split into feature matrix (X) and target vector (y)
X = df_clean[['age', 'company_encoded']]
y = df_clean['gender'].map({'male': 0, 'female': 1}) # 0: Male, 1: Female

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🏋️‍♂️ Fitting RandomForest Classifier on traveler demographic signatures...")
classifier = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
classifier.fit(X_train, y_train)

# Performance Evaluation
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n=========================================")
print(f"✅ Demographic Classification Complete!")
print(f"📊 Accuracy Score: {accuracy * 100:.2f}%")
print("=========================================")
print(classification_report(y_test, y_pred, target_names=['male', 'female']))

# Serialize and save all pipeline components safely
joblib.dump(classifier, os.path.join(MODEL_DIR, "rf_classifier.pkl"))
joblib.dump(company_encoder, os.path.join(MODEL_DIR, "company_encoder.pkl"))
print("💾 Classification model and encoder saved successfully in Model/ directory.")