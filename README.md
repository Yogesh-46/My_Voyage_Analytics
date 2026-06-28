# ✈️ Voyage Analytics Intelligence Engine

A production-oriented Machine Learning and MLOps project for the Travel & Tourism domain.

This project predicts flight prices, classifies traveler gender, and recommends hotels using machine learning models exposed through a Flask REST API and visualized with a Streamlit dashboard. The project also demonstrates modern MLOps practices including MLflow, Apache Airflow, Jenkins, Docker, and Kubernetes.

---

## Business Problem

Travel companies manage large volumes of customer, flight, and hotel data. This project demonstrates how machine learning can improve travel decision-making by providing:

- Flight price prediction
- Traveler gender classification
- Personalized hotel recommendations

The project also focuses on deploying and managing ML models using production-oriented MLOps practices.

---

# Features

✅ Flight Price Prediction (Regression)

- XGBoost Regression Model
- REST API using Flask
- Real-time predictions

---

✅ Gender Classification

- Random Forest Classifier
- Company, Age and Name-based feature engineering
- Improved model accuracy through feature engineering

---

✅ Hotel Recommendation

- Collaborative Filtering
- Personalized hotel suggestions
- Popularity fallback for new users

---

✅ Streamlit Dashboard

Interactive dashboard supporting:

- Flight Price Prediction
- Gender Classification
- Hotel Recommendation

---

✅ MLOps Components

- Flask REST API
- MLflow Experiment Tracking
- Apache Airflow DAGs
- Jenkins CI/CD
- Docker Containerization
- Kubernetes Deployment

---

# System Architecture

```
                    Streamlit Dashboard
                           │
                           ▼
                    Flask REST API
          ┌───────────┬────────────┬────────────┐
          ▼           ▼            ▼
     Regression   Classification  Recommendation
          │           │            │
          └───────────┴────────────┘
                     ML Models
                         │
                     MLflow Tracking
                         │
         Docker → Kubernetes Deployment
                         │
                  Jenkins CI/CD Pipeline
                         │
                Apache Airflow Workflows
```

---

# Dataset

The project uses three datasets.

### Users

- User ID
- Name
- Company
- Gender
- Age

### Flights

- Flight Route
- Agency
- Flight Type
- Price
- Distance
- Time

### Hotels

- Hotel Name
- Destination
- Price
- Stay Duration

---

# Machine Learning Models

## Flight Price Prediction

Algorithm

- XGBoost Regressor

Features

- Origin
- Destination
- Agency
- Flight Type

Output

- Predicted Flight Price

---

## Gender Classification

Algorithm

- Random Forest

Features

- Age
- Company
- First Name (Feature Engineered)

Output

- Male / Female

---

## Hotel Recommendation

Algorithm

- Collaborative Filtering

Output

- Top Recommended Hotels

---

# REST API

## Flight Prediction

POST

```
/predict
```

Example Request

```json
{
    "from":"Sao Paulo (SP)",
    "to":"Rio de Janeiro (RJ)",
    "agency":"CloudFy",
    "flightType":"economic"
}
```

---

## Gender Classification

POST

```
/classify_user
```

Example Request

```json
{
    "age":25,
    "company":"4You",
    "name":"Roy Braun"
}
```

---

## Hotel Recommendation

POST

```
/recommend_hotels
```

Example Request

```json
{
    "userCode":15,
    "destination_city":"Rio de Janeiro (RJ)"
}
```

---

# Project Structure

```
App/
│
├── price_predictor.py

Data/
│
├── Flight_Prediction/
├── Users/
├── Hotels/

Model/
│
├── flight_prediction/
├── flight_classification/
├── hotel_recommendation/

Src/
│
├── Flight_Prediction/
├── flight_classification/
├── hotel_recommendation/

dags/

cicd/

Dockerfile

deployment.yaml

requirements.txt

README.md
```

---

# Technology Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Flask
- Streamlit
- MLflow
- Apache Airflow
- Docker
- Kubernetes
- Jenkins

---

# Running the Project

Install dependencies

```bash
pip install -r requirements.txt
```

Run Flask API

```bash
python Src/Flight_Prediction/api.py
```

Run Streamlit Dashboard

```bash
streamlit run App/price_predictor.py
```

---

# Docker

A Dockerfile has been provided for containerized deployment.

Due to development environment restrictions on a managed corporate laptop, the project was developed and validated using a Python virtual environment. The Docker configuration is included to demonstrate containerization for deployment.

---

# MLOps Workflow

1. Train Machine Learning Models
2. Track Experiments using MLflow
3. Automate Workflows using Airflow
4. Build Docker Image
5. Deploy using Kubernetes
6. CI/CD using Jenkins
7. Serve Predictions using Flask
8. Visualize Results using Streamlit

---

# Future Improvements

- Model Monitoring
- Automated Retraining
- User Authentication
- Cloud Deployment
- Improved Recommendation Engine

---

# Author

**Yogesh Dubey**

BI Engineer | Data Science & AI Enthusiast

GitHub: https://github.com/Yogesh-46