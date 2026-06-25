# ✈️ Voyage Analytics: Intelligent Flight Pricing & MLOps Platform

Voyage Analytics is an enterprise-grade, end-to-end MLOps platform built to predict real-world flight pricing distributions. Utilizing the **Argo Datathon Travel Dataset**, the system features a decoupled microservices architecture, automated model training pipelines, containerized deployment workflows, and robust experiment tracking.

---

## 🏛️ System Architecture Overview

The platform is engineered using a decoupled, production-ready microservices layout:
1. **Frontend UI (Streamlit):** A wide-screen responsive web dashboard providing real-time flight configuration inputs and visual metric displays.
2. **Backend REST API (Flask):** An independent inference server that serves model predictions over HTTP via structured JSON payloads.
3. **Machine Learning Engine (XGBoost):** An optimized gradient boosting regressor hitting an **80.23% R² predictive accuracy**.
4. **MLOps Lifecycle Layer (MLflow, Airflow, Jenkins, Docker, Kubernetes):** Automates tracking, containerization, orchestration, and continuous integration.

---

## 🗂️ Repository Directory Structure

```text
my_voyage_analytics/
├── app/
│   └── price_predictor.py       # Streamlit Web Application
├── cicd/
│   └── Jenkinsfile             # Jenkins Continuous Integration Pipeline Script
├── dags/
│   └── flight_retraining_dag.py # Apache Airflow Automated Orchestration Workflow
├── data/
│   └── flight_prediction/
│       ├── flights.csv          # Raw Travel Logs
│       └── processed_flights.csv# Clean, Feature-Engineered Dataset
├── model/
│   └── flight_prediction/       # Serialized Model Artifacts & Encoders (.pkl)
├── src/
│   └── flight_prediction/
│       ├── preprocessing.py     # Target Encoding & Data Cleaning Pipeline
│       ├── train.py             # XGBoost Model Training & MLflow Logging
│       └── api.py               # Flask REST API Microservice Engine
├── Dockerfile                   # Deployment Container Blueprint
├── deployment.yaml              # Kubernetes Deployment & Service Manifests
└── requirements.txt             # Production Library Dependencies Shopping List