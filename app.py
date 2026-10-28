import os
from datetime import datetime
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "sugarcane_readings.csv")
os.makedirs(DATA_DIR, exist_ok=True)

COLUMNS = [
    "timestamp", "temperature_c", "humidity_percent", "soil_moisture_percent",
    "brix", "crop_age_days", "variety", "maturity_label", "notes"
]

st.set_page_config(page_title="Sugarcane Maturity Prediction", page_icon="🌱", layout="wide")
st.title("🌱 Sugarcane Maturity Prediction Device")
st.caption("A farmer-friendly prototype for recording field readings and exploring maturity data.")
st.warning("Prototype only: predictions require suitable, verified field data. Do not use demo or guessed labels as scientific evidence.")

if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=COLUMNS).to_csv(DATA_FILE, index=False)

df = pd.read_csv(DATA_FILE)
tab_dashboard, tab_add, tab_data, tab_model = st.tabs(["Dashboard", "Add Field Reading", "Dataset", "Model"])

with tab_dashboard:
    st.subheader("Field Monitoring Dashboard")
    a, b, c = st.columns(3)
    a.metric("Saved readings", len(df))
    if len(df):
        latest = df.iloc[-1]
        b.metric("Latest temperature", f"{latest['temperature_c']} °C" if pd.notna(latest['temperature_c']) else "—")
        c.metric("Latest humidity", f"{latest['humidity_percent']} %" if pd.notna(latest['humidity_percent']) else "—")
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        b.metric("Latest temperature", "—")
        c.metric("Latest humidity", "—")
        st.info("No readings saved yet. Open **Add Field Reading** to enter values from your ESP32 display and refractometer.")

with tab_add:
    st.subheader("Record a field sample")
    with st.form("reading_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=70.0, value=28.0, step=0.1)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
            soil = st.number_input("Soil moisture (%)", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
            brix = st.number_input("Brix (manual refractometer reading)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
        with col2:
            crop_age = st.number_input("Crop age (days, if known)", min_value=0, max_value=1000, value=0, step=1)
            variety = st.text_input("Sugarcane variety (if known)")
            maturity = st.selectbox("Verified maturity label (optional)", ["Not verified", "Immature", "Semi-Mature", "Mature"])
            notes = st.text_area("Notes / sample location")
        submitted = st.form_submit_button("Save reading")
        
if submitted:
    # DEMO RULE: illustrative thresholds only, not scientifically validated
    BRIX_THRESHOLD = 18.0
    AGE_THRESHOLD_DAYS = 300

    if brix > 0 and crop_age > 0:
        if brix >= BRIX_THRESHOLD and crop_age >= AGE_THRESHOLD_DAYS:
            result = "LIKELY MATURE"
        else:
            result = "LIKELY IMMATURE / NEEDS MORE CHECKING"
    else:
        result = "CANNOT ASSESS — enter Brix and crop age"

    st.subheader("Sugarcane Maturity Result")
    if result == "LIKELY MATURE":
        st.success("🌾 " + result)
    elif result.startswith("LIKELY IMMATURE"):
        st.warning("🌱 " + result)
    else:
        st.info(result)

    st.caption(
        "Expo demonstration only. Thresholds are illustrative and "
        "must be validated for the sugarcane variety and growing region."
    )

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "temperature_c": temperature,
        "humidity_percent": humidity,
        "soil_moisture_percent": soil,
        "brix": brix if brix > 0 else None,
        "crop_age_days": crop_age if crop_age > 0 else None,
        "variety": variety.strip(),
        "maturity_label": result,
        "notes": notes.strip()
    }

    updated = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(DATA_FILE, index=False)
    st.success("Reading saved.")


with tab_data:
    st.subheader("Recorded dataset")
    st.caption("Only enter actual measurements. Labels should be verified independently; don't invent maturity labels.")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV dataset", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="sugarcane_readings.csv", mime="text/csv")

with tab_model:
    st.subheader("Machine-learning model")
    st.write("This page checks whether there is enough verified data to train a baseline classifier.")
    st.info("Model training is not run automatically. Add real readings with independently verified maturity labels, then use train_model.py in a Python environment.")
    if "maturity_label" in df and len(df):
        labelled = df[df["maturity_label"].isin(["Immature", "Semi-Mature", "Mature"])]
        st.write(f"Rows with verified maturity labels: **{len(labelled)}**")
        if len(labelled):
            st.write(labelled["maturity_label"].value_counts().rename_axis("label").reset_index(name="count"))
        else:
            st.caption("No verified labels have been entered yet.")
