
import os
from datetime import datetime

import pandas as pd
import streamlit as st

# -----------------------------
# Configuration
# -----------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "sugarcane_readings.csv")

os.makedirs(DATA_DIR, exist_ok=True)

COLUMNS = [
    "timestamp",
    "temperature_c",
    "humidity_percent",
    "soil_moisture_percent",
    "brix",
    "crop_age_days",
    "variety",
    "maturity_label",
    "notes",
]

st.set_page_config(
    page_title="Sugarcane Maturity Prediction",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 Sugarcane Maturity Prediction Device")
st.caption(
    "A farmer-friendly prototype for recording field readings "
    "and assessing sugarcane maturity."
)

st.warning(
    "Expo demonstration: maturity results use illustrative rules, "
    "not a trained or scientifically validated prediction model."
)

# -----------------------------
# Dataset initialization
# -----------------------------
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=COLUMNS).to_csv(DATA_FILE, index=False)


def load_data():
    data = pd.read_csv(DATA_FILE)

    # Ensure older CSV files have all required columns.
    for column in COLUMNS:
        if column not in data.columns:
            data[column] = None

    return data[COLUMNS]


# -----------------------------
# Navigation tabs
# -----------------------------
tab_dashboard, tab_check, tab_dataset, tab_model = st.tabs(
    ["Dashboard", "Check Maturity", "Dataset", "Model Information"]
)

# -----------------------------
# Dashboard
# -----------------------------
with tab_dashboard:
    df = load_data()

    st.subheader("Field Monitoring Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Saved Samples", len(df))

    if not df.empty:
        latest = df.iloc[-1]

        col2.metric(
            "Temperature",
            f"{latest['temperature_c']} °C"
            if pd.notna(latest["temperature_c"])
            else "—",
        )

        col3.metric(
            "Humidity",
            f"{latest['humidity_percent']}%"
            if pd.notna(latest["humidity_percent"])
            else "—",
        )

        col4.metric(
            "Latest Brix",
            f"{latest['brix']}"
            if pd.notna(latest["brix"])
            else "—",
        )

        st.subheader("Recent Field Readings")
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        col2.metric("Temperature", "—")
        col3.metric("Humidity", "—")
        col4.metric("Latest Brix", "—")

        st.info("No readings saved yet. Open Check Maturity to begin.")

# -----------------------------
# Check maturity
# -----------------------------
with tab_check:
    st.subheader("Enter Sugarcane Sample Details")

    st.write(
        "Enter the measurements from your sensors and refractometer "
        "to obtain a demonstration maturity assessment."
    )

    with st.form("maturity_form"):
        left, right = st.columns(2)

        with left:
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-10.0,
                max_value=70.0,
                value=28.0,
                step=0.1,
            )

            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=1.0,
            )

            soil = st.number_input(
                "Soil Moisture (%)",
                min_value=0.0,
                max_value=100.0,
                value=40.0,
                step=1.0,
            )

            brix = st.number_input(
                "Brix Reading",
                min_value=0.0,
                max_value=50.0,
                value=16.0,
                step=0.5,
                help="Enter the actual reading from your refractometer.",
            )

        with right:
            crop_age = st.number_input(
                "Crop Age (days)",
                min_value=0,
                max_value=1000,
                value=300,
                step=1,
            )

            variety = st.text_input(
                "Sugarcane Variety (optional)"
            )

            notes = st.text_area(
                "Sample Location / Notes (optional)"
            )

        st.markdown("#### Demonstration Thresholds")

        threshold_col1, threshold_col2 = st.columns(2)

        with threshold_col1:
            brix_threshold = st.number_input(
                "Brix Threshold",
                min_value=0.0,
                max_value=50.0,
                value=18.0,
                step=0.5,
            )

        with threshold_col2:
            age_threshold = st.number_input(
                "Crop Age Threshold (days)",
                min_value=0,
                max_value=1000,
                value=300,
                step=10,
            )

        submitted = st.form_submit_button(
            "Check Maturity and Save",
            type="primary",
        )

    if submitted:
        # Simple rule-based demonstration.
        # These thresholds are illustrative, not validated standards.
        if brix <= 0 or crop_age <= 0:
            result = "CANNOT ASSESS"
            explanation = (
                "Enter a positive Brix reading and crop age "
                "to obtain the demonstration result."
            )

        elif brix >= brix_threshold and crop_age >= age_threshold:
            result = "LIKELY MATURE"
            explanation = (
                "The entered Brix and crop age meet both "
                "demonstration thresholds."
            )

        else:
            result = "LIKELY IMMATURE / NEEDS MORE CHECKING"
            explanation = (
                "At least one entered value is below its "
                "demonstration threshold."
            )

        st.subheader("Maturity Assessment")

        if result == "LIKELY MATURE":
            st.success("🌾 " + result)
        elif result == "CANNOT ASSESS":
            st.info("ℹ️ " + result)
        else:
            st.warning("🌱 " + result)

        st.write(explanation)

        st.write(f"**Brix entered:** {brix}")
        st.write(f"**Crop age entered:** {crop_age} days")

        st.caption(
            "Important: this is a rule-based expo demonstration, "
            "not a trained ML prediction or a reliable harvest recommendation. "
            "Thresholds must be validated for the local variety and growing region."
        )

        # Save the actual entered sample and its demo result.
        new_row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "temperature_c": temperature,
            "humidity_percent": humidity,
            "soil_moisture_percent": soil,
            "brix": brix,
            "crop_age_days": crop_age,
            "variety": variety.strip(),
            "maturity_label": result,
            "notes": notes.strip(),
        }

        current_data = load_data()
        updated_data = pd.concat(
            [current_data, pd.DataFrame([new_row])],
            ignore_index=True,
        )

        updated_data.to_csv(DATA_FILE, index=False)

        st.success("Sample saved to the dataset.")

# -----------------------------
# Dataset
# -----------------------------
with tab_dataset:
    st.subheader("Sugarcane Dataset")

    df = load_data()

    st.write(f"Total saved samples: **{len(df)}**")

    if df.empty:
        st.info("No samples have been saved yet.")
    else:
        st.dataframe(df, use_container_width=True)

    st.download_button(
        label="Download Dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="sugarcane_readings.csv",
        mime="text/csv",
    )

# -----------------------------
# Model information
# -----------------------------
with tab_model:
    st.subheader("Machine Learning Information")

    st.write(
        "The intended machine-learning approach is a "
        "**Random Forest Classifier**."
    )

    st.write(
        "A trained model requires real field readings and independently "
        "verified maturity labels. This version of the website does not "
        "use a trained Random Forest model to produce its maturity result."
    )

    st.markdown(
        """
        **Required future steps:**

        1. Collect representative sugarcane field samples.
        2. Measure Brix and record crop and environmental information.
        3. Obtain independently verified maturity labels.
        4. Train the classifier using labelled data.
        5. Evaluate it on held-out samples.
        6. Integrate the trained model into the website.
        """
    )

st.divider()

st.caption(
    "Sugarcane Maturity Prediction Device | Project Expo Prototype"
)
