import streamlit as st
import requests
import pandas as pd

# 1. Config page & inject high-end enterprise CSS styling
st.set_page_config(page_title="Voyage Analytics Core", layout="wide", page_icon="✈️")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    div[data-testid="stMetricValue"] { font-size: 2.4rem !important; font-weight: 700 !important; color: #58a6ff !important; }
    .report-card { background: linear-gradient(135deg, #161b22 0%, #21262d 100%); padding: 24px; border-radius: 12px; border: 1px solid #30363d; box-shadow: 0 4px 12px rgba(0,0,0,0.3); margin-bottom: 15px; }
    .card-title { font-size: 1.25rem; font-weight: 600; color: #f0f6fc; margin-bottom: 15px; display: flex; align-items: center; gap: 8px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='background: linear-gradient(90deg, #1f6feb 0%, #112a59 100%); padding: 25px; border-radius: 10px; border: 1px solid #388bfd; margin-bottom: 25px;'>
        <h1 style='margin:0; color: white; font-family: system-ui; font-weight: 800;'>🛰️ VOYAGE ANALYTICS INTELLIGENCE ENGINE</h1>
        <p style='margin:5px 0 0 0; color: #8bc2ff; font-size: 1.05rem;'>Multi-Model Predictive Control Tower • Production API v2.4.0</p>
    </div>
""", unsafe_allow_html=True)

BASE_URL = "http://127.0.0.1:5001"

@st.cache_data
def load_data():
    flights = pd.read_csv("Data/Flight_Prediction/flights.csv")
    users = pd.read_csv("Data/Users/users.csv")
    return flights, users

flights_df, users_df = load_data()

FROM_CITIES = sorted(flights_df["from"].unique())
TO_CITIES = sorted(flights_df["to"].unique())
AGENCIES = sorted(flights_df["agency"].unique())
FLIGHT_TYPES = sorted(flights_df["flightType"].unique())
COMPANIES = sorted(users_df["company"].unique())

# 3. Sidebar Navigation for Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/airplane-take-off.png", width=70)
    st.markdown("### 🎛️ Control & Parameters")
    st.markdown("---")
    
    st.markdown("#### **Route Vectors**")
    from_city = st.selectbox("Departure Hub (From)", FROM_CITIES, index=0)
    default_to_idx = 1 if len(TO_CITIES) > 1 else 0
    to_city = st.selectbox("Destination Hub (To)", TO_CITIES, index=default_to_idx)
    
    # 📅 NEW: Interactive Temporal Selection Grid
    st.markdown("#### **Temporal Matrix**")
    travel_date = st.date_input("Select Departure Date", value=pd.to_datetime("2022-06-15"))
    
    st.markdown("#### **Logistics Attributes**")
    agency = st.selectbox("Booking Pipeline Provider", AGENCIES)
    flight_type = st.selectbox("Seat Inventory Tier", FLIGHT_TYPES)
    
    st.markdown("#### **Client Signature Vector**")
    user_code = st.number_input("User Identification Code (User ID)", min_value=0, max_value=1339, value=0, step=1)
    age = st.slider("Traveler Age", min_value=18, max_value=80, value=35)
    company = st.selectbox("Company", COMPANIES)
    
    st.markdown("---")
    run_analysis = st.button("🚀 Execute Matrix Inference", use_container_width=True)

# 4. Primary Core Dashboard Output Workspace
if run_analysis:
    if from_city == to_city:
        st.error("🚨 **Routing Fault Matrix Exception:** The designated Departure Hub and Destination Hub cannot map to identical vector nodes. Please reset your inputs.")
    else:
        res_col1, res_col2, res_col3 = st.columns(3)
        
        # 📅 Extract Day, Month, and Year directly from the selected date object
        day_val = travel_date.day
        month_val = travel_date.month
        year_val = travel_date.year
        
        # ----------------------------------------------------
        # COLUMN 1: FLIGHT VALUATION GRAPH CARD
        # ----------------------------------------------------
        with res_col1:
            st.markdown("<div class='report-card'><div class='card-title'>🎚️ Ticket Price Valuation</div></div>", unsafe_allow_html=True)
            
            # Map parameters alongside our newly parsed calendar metrics
            payload_reg = {
                "from": from_city, 
                "to": to_city, 
                "agency": agency, 
                "flightType": flight_type,
                "day": int(day_val),
                "month": int(month_val),
                "year": int(year_val)
            }
            try:
                response_reg = requests.post(f"{BASE_URL}/predict", json=payload_reg)
                if response_reg.status_code == 200:
                    predicted_price = response_reg.json().get("predicted_price", 0.0)
                    st.metric(label="Calculated Clean Fare Matrix Value", value=f"${predicted_price:,.2f}")
                    st.success("⚡ XGBoost Hyper-Optimized Inference Stable")
                else:
                    st.error("Inference cluster failed to return target matrix vector.")
            except Exception as e:
                st.error(f"Gateway connection drop: {e}")

        # ----------------------------------------------------
        # COLUMN 2: CUSTOMER SEGMENT MATRIX CARD
        # ----------------------------------------------------
        with res_col2:
            st.markdown("<div class='report-card'><div class='card-title'>👥 Traveler Segmentation Engine</div></div>", unsafe_allow_html=True)
            
            if "userCode" in users_df.columns:
                matched_user = users_df[users_df["userCode"] == user_code]
                resolved_name = matched_user["name"].values[0] if not matched_user.empty else "Unknown"
            else:
                resolved_name = users_df.iloc[user_code]["name"] if user_code < len(users_df) else "Unknown"
                
            payload_clf = {"age": int(age), "company": company, "name": resolved_name}
            try:
                response_clf = requests.post(f"{BASE_URL}/classify_user", json=payload_clf)
                if response_clf.status_code == 200:
                    predicted_gender = response_clf.json().get("predicted_gender", "Unknown")
                    
                    if age < 30:
                        seniority, archetype = "Associate Specialist", "Young Professional Segment"
                    elif 30 <= age <= 50:
                        seniority, archetype = "Regional Executive", "Mid-Career Corporate Cluster"
                    else:
                        seniority, archetype = "Managing Director", "Senior Enterprise Executive"
                        
                    st.metric(label="Predicted Gender Domain Class", value=predicted_gender)
                    st.markdown(f"**Identified Operational Profile:**")
                    st.info(f"🧬 **{archetype}**\n\n🎯 *Imputed Seniority Rank:* `{seniority}`\n\n🏢 *Corporate Account:* {company}")
                else:
                    st.error("Demographic telemetry classification timeout failure.")
            except Exception as e:
                st.error(f"Gateway connection drop: {e}")

        # ----------------------------------------------------
        # COLUMN 3: HOTEL MATRICES COLLABORATIVE CARD
        # ----------------------------------------------------
        with res_col3:
            st.markdown("<div class='report-card'><div class='card-title'>🏨 Hospitality Matrix Matcher</div></div>", unsafe_allow_html=True)
            payload_rec = {"userCode": int(user_code), "destination_city": to_city}
            try:
                response_rec = requests.post(f"{BASE_URL}/recommend_hotels", json=payload_rec)
                if response_rec.status_code == 200:
                    data_rec = response_rec.json()
                    recommendations = data_rec.get("recommendations", [])
                    rec_type = data_rec.get("type", "Standard")
                    
                    st.markdown(f"**Top Recommendations ({rec_type}):**")
                    if recommendations:
                        for idx, hotel in enumerate(recommendations, start=1):
                            st.markdown(f"""
                                <div style='background-color:#21262d; padding:10px; border-radius:6px; border-left:4px solid #58a6ff; margin-bottom:8px;'>
                                    <span style='color:#f0f6fc; font-weight:600;'>{idx}️. {hotel}</span>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.write("No hospitality coordinates match this target city hub.")
                else:
                    st.error("Matrix similarity calculation fault.")
            except Exception as e:
                st.error(f"Gateway connection drop: {e}")
else:
    st.markdown("""
        <div style='text-align: center; padding: 60px; background-color: #161b22; border: 1px dashed #30363d; border-radius: 12px; margin-top: 20px;'>
            <img src="https://img.icons8.com/fluency/96/radar.png" style="opacity: 0.7; margin-bottom: 15px; width: 64px;">
            <h3 style='color: #8b949e; font-weight: 500;'>Awaiting Parameter Dispatch Pipeline Input</h3>
            <p style='color: #484f58; max-width: 500px; margin: 0 auto; font-size: 0.95rem;'>Configure traveler profiling values and route paths in the control panel sidebar, then click 'Execute Matrix Inference' to start real-time prediction pipelines.</p>
        </div>
    """, unsafe_allow_html=True)