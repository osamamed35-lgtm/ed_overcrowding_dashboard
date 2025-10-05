import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import datetime

# ----------------------------
# Train a base logistic regression model (synthetic)
# ----------------------------
np.random.seed(42)
n = 300
patients_waiting = np.random.randint(20, 150, n)
available_staff = np.random.randint(3, 20, n)
avg_consult_time = np.random.randint(5, 30, n)
incoming_ambulances = np.random.randint(0, 10, n)
free_beds = np.random.randint(0, 15, n)

# underlying overcrowding logic
z = (0.04 * patients_waiting 
     - 0.25 * available_staff 
     + 0.08 * avg_consult_time 
     + 0.10 * incoming_ambulances 
     - 0.05 * free_beds - 5)
prob = 1 / (1 + np.exp(-z))
y = (prob > 0.5).astype(int)

X = np.c_[patients_waiting, available_staff, avg_consult_time, incoming_ambulances, free_beds]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(C=0.5, max_iter=5000)
model.fit(X_scaled, y)

# ----------------------------
# Streamlit Layout
# ----------------------------
st.set_page_config(page_title="ED Overcrowding Dashboard", layout="wide")
st.title("🏥 ED Overcrowding Command Dashboard")
st.markdown("""
Monitor and predict Emergency Department (ED) overcrowding risk in real time.  
Upload hospital data for analytics, or use the left panel for what-if simulations.
""")

# Sidebar Inputs
st.sidebar.header("⚙️ Manual Input (What-if Analysis)")
patients = st.sidebar.slider("Patients waiting", 20, 150, 80)
staff = st.sidebar.slider("Available staff", 3, 20, 10)
consult_time = st.sidebar.slider("Average consult time (min)", 5, 30, 15)
ambulances = st.sidebar.slider("Incoming ambulances (next 2h)", 0, 10, 3)
beds = st.sidebar.slider("Free beds", 0, 15, 5)

# ----------------------------
# Live Single Prediction
# ----------------------------
user_data = np.array([[patients, staff, consult_time, ambulances, beds]])
user_scaled = scaler.transform(user_data)
probability = model.predict_proba(user_scaled)[0, 1]

col1, col2, col3 = st.columns(3)
col1.metric("Patients", patients)
col2.metric("Staff on duty", staff)
import plotly.graph_objects as go

# Create a circular gauge for ED Pressure
import plotly.graph_objects as go
# Calculate pressure percentage from model prediction
pressure = probability * 100
col1, col2, col3 = st.columns(3)
col1.metric("Patients", patients)
col2.metric("Staff on duty", staff)
col3.metric("🏥 ED Pressure Level", f"{probability*100:.1f}%")

# Calculate pressure percentage for gauge
pressure = probability * 100

import plotly.graph_objects as go

# Create a circular gauge for ED Pressure
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=pressure,
    number={'suffix': "%"},
    title={'text': "ED Pressure Level", 'font': {'size': 20}},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 40], 'color': "green"},
            {'range': [40, 60], 'color': "yellow"},
            {'range': [60, 80], 'color': "orange"},
            {'range': [80, 100], 'color': "red"}
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': pressure
        }
    }
))

st.plotly_chart(fig_gauge, use_container_width=True)
# Show a friendly and color-coded “pressure level”
pressure = probability * 100
col3.metric("🏥 ED Pressure Level", f"{pressure:.1f} %")

if probability >= 0.8:
    st.markdown("### 🟥 **Critical Pressure!** — Immediate surge response required.")
elif probability >= 0.6:
    st.markdown("### 🟧 **High Pressure!** — Prepare to activate contingency plans.")
elif probability >= 0.4:
    st.markdown("### 🟨 **Moderate Pressure** — Monitor and redistribute load.")
else:
    st.markdown("### 🟩 **Stable Operations** — ED flow is under control.")
# ----------------------------
# File Upload Section
# ----------------------------
st.markdown("### 📤 Upload Historical ED Data (CSV)")
uploaded_file = st.file_uploader(
    "Upload CSV with columns: Patients, Staff, ConsultTime, Ambulances, Beds, Timestamp (optional)",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    expected_cols = ["Patients", "Staff", "ConsultTime", "Ambulances", "Beds"]
    for col in expected_cols:
        if col not in data.columns:
            st.error(f"Missing column: {col}")
            st.stop()

    # Predictions
    X_new = data[expected_cols].values
    X_new_scaled = scaler.transform(X_new)
    data["PredictedProb"] = model.predict_proba(X_new_scaled)[:, 1]
    data["PredictedLabel"] = (data["PredictedProb"] > 0.5).astype(int)

    if "Timestamp" not in data.columns:
        data["Timestamp"] = pd.date_range(end=datetime.datetime.now(), periods=len(data), freq="H")

    st.success(f"✅ File loaded successfully — {len(data)} records analyzed.")

    # ----------------------------
    # 🔹 Trend Summary Section
    # ----------------------------
    avg_risk = data["PredictedProb"].mean()
    max_risk_row = data.loc[data["PredictedProb"].idxmax()]
    min_risk_row = data.loc[data["PredictedProb"].idxmin()]

    st.markdown("### 📊 Overcrowding Trend Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Risk", f"{avg_risk*100:.1f} %")
    c2.metric("Highest Risk Time", 
              f"{max_risk_row['Timestamp']:%Y-%m-%d %H:%M}",
              f"{max_risk_row['PredictedProb']*100:.1f}%")
    c3.metric("Safest Period",
              f"{min_risk_row['Timestamp']:%Y-%m-%d %H:%M}",
              f"{min_risk_row['PredictedProb']*100:.1f}%")

    # ----------------------------
    # 🔹 Risk Trend Visualization
    # ----------------------------
    st.markdown("### 📈 Overcrowding Risk Over Time")
    fig = px.line(data, x="Timestamp", y="PredictedProb",
                  color="PredictedLabel",
                  color_discrete_map={0: "green", 1: "red"},
                  labels={"PredictedProb": "P(Overcrowded)"},
                  title="Hourly Overcrowding Probability")
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # 🔹 Feature Insights
    # ----------------------------
    st.markdown("### 🔍 Average Feature Levels (in Uploaded Data)")
    mean_vals = data[expected_cols].mean().reset_index()
    mean_vals.columns = ["Feature", "Average"]
    bar = px.bar(mean_vals, x="Feature", y="Average", color="Average",
                 color_continuous_scale="Blues",
                 title="Average Levels of Key Operational Indicators")
    st.plotly_chart(bar, use_container_width=True)

    # ----------------------------
    # 🔹 Raw Data Viewer
    # ----------------------------
    with st.expander("📋 View Data Table"):
        st.dataframe(data.tail(20))
else:
    st.info("Upload a CSV file to analyze historical ED overcrowding trends.")