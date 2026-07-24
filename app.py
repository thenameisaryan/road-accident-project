import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Accident Severity Risk Evaluator",
    page_icon="🚗",
    layout="wide"
)

# Load Artifacts
@st.cache_resource
def load_package():
    return joblib.load("accident_severity_model.pkl")

try:
    pkg = load_package()
    model = pkg["model"]
    encoder = pkg["encoder"]
    categories = pkg["categories"]
    s_min = pkg.get("s_min", 0.08)
    s_max = pkg.get("s_max", 0.35)
except Exception:
    st.error("⚠️ Model file 'accident_severity_model.pkl' not found. Run 'train_model.py' first.")
    st.stop()

# Dynamic Risk Mapper
def compute_risk_profile(raw_s, s_min, s_max):
    # Map raw model output to relative risk index (0.0 = safest, 1.0 = most hazardous)
    risk_index = float(np.clip((raw_s - s_min) / (s_max - s_min + 1e-6), 0.0, 1.0))
    
    # Calculate non-linear probability response curves
    p_fatal = 0.01 + 0.34 * (risk_index ** 1.8)
    p_serious = 0.10 + 0.35 * risk_index
    p_slight = max(0.01, 1.0 - p_fatal - p_serious)
    
    total = p_slight + p_serious + p_fatal
    return risk_index, p_slight / total, p_serious / total, p_fatal / total

st.title("🚗 Road Accident Severity Risk Evaluator")
st.caption("Dynamic risk evaluation with monotonic gradient boosting constraints.")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("⏱️ Primary Factors")
    speed_limit = st.select_slider("Speed Limit (mph)", options=[20, 30, 40, 50, 60, 70], value=30)
    number_of_vehicles = st.number_input("Vehicles Involved", min_value=1, max_value=15, value=2)
    accident_hour = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=18)

with col2:
    st.subheader("🌧️ Lighting & Weather")
    light_conditions = st.selectbox("Light Conditions", categories["light_order"])
    weather_conditions = st.selectbox("Weather Conditions", categories["weather_order"])

with col3:
    st.subheader("🛣️ Road & Area")
    road_surface = st.selectbox("Road Surface Conditions", categories["surface_order"])
    urban_rural = st.selectbox("Area Type", categories["area_order"])

st.markdown("---")

if st.button("📊 Evaluate Severity Risk", use_container_width=True):
    # Encode user selections
    cat_df = pd.DataFrame([{
        "Light_Conditions": light_conditions,
        "Weather_Conditions": weather_conditions,
        "Road_Surface_Conditions": road_surface,
        "Urban_or_Rural_Area": urban_rural
    }])
    
    encoded_cats = encoder.transform(cat_df)
    
    input_data = pd.DataFrame([{
        "Speed_limit": float(speed_limit),
        "Number_of_Vehicles": float(number_of_vehicles),
        "Accident_Hour": float(accident_hour),
        "Light_Conditions_code": encoded_cats[0][0],
        "Weather_Conditions_code": encoded_cats[0][1],
        "Road_Surface_Conditions_code": encoded_cats[0][2],
        "Urban_or_Rural_Area_code": encoded_cats[0][3]
    }])
    
    # Predict raw severity index S
    raw_severity = model.predict(input_data)[0]
    risk_index, p_slight, p_serious, p_fatal = compute_risk_profile(raw_severity, s_min, s_max)

    st.subheader("🎯 Risk Assessment Analysis")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric(
            label="Relative Risk Index",
            value=f"{risk_index * 100:.1f}%",
            delta=f"Raw Severity: {raw_severity:.3f}"
        )

        if risk_index >= 0.70 or p_fatal >= 0.20:
            st.error("🚨 **Predicted Risk Level: High Severity (Fatal Risk Elevated)**")
        elif risk_index >= 0.35 or p_serious >= 0.25:
            st.warning("⚠️ **Predicted Risk Level: Moderate / Serious**")
        else:
            st.success("✅ **Predicted Risk Level: Low / Slight**")

    with res_col2:
        st.markdown("#### Probability Distribution")

        st.write(f"**Fatal Risk:** `{p_fatal * 100:.1f}%`")
        st.progress(float(p_fatal))

        st.write(f"**Serious Risk:** `{p_serious * 100:.1f}%`")
        st.progress(float(p_serious))

        st.write(f"**Slight Risk:** `{p_slight * 100:.1f}%`")
        st.progress(float(p_slight))
