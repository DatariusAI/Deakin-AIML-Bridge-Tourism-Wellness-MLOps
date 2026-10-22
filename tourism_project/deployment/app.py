"""Streamlit app: predict whether a customer will buy the Wellness Tourism Package.

Streamlit Community Cloud runs this file from the repo root, so paths are
resolved relative to this file, not the working directory.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "best_tourism_model_v1.joblib"
META_PATH = HERE / "model_metadata.json"

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧳", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}


model = load_model()
meta = load_metadata()
threshold = float(meta.get("threshold", 0.5))

st.title("Visit with Us: Wellness Tourism Package")
st.write(
    "Enter the customer and sales-pitch details. The model estimates the chance "
    "that this customer buys the Wellness Tourism Package, so the sales team can "
    "prioritise who to contact."
)

with st.form("customer_form"):
    st.subheader("Customer details")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=90, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital status", ["Married", "Single", "Unmarried", "Divorced"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly income", min_value=0.0, max_value=100000.0, value=22000.0, step=500.0)
        city_tier = st.selectbox("City tier", [1, 2, 3])
    with c2:
        type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
        num_person = st.number_input("Number of persons visiting", min_value=1, max_value=10, value=3)
        num_children = st.number_input("Children (under 5) visiting", min_value=0, max_value=5, value=1)
        num_trips = st.number_input("Trips per year", min_value=0, max_value=25, value=3)
        property_star = st.selectbox("Preferred property star", [3, 4, 5])
        passport = st.selectbox("Has passport", ["Yes", "No"])
        own_car = st.selectbox("Owns a car", ["Yes", "No"])

    st.subheader("Sales interaction")
    c3, c4 = st.columns(2)
    with c3:
        product_pitched = st.selectbox("Product pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
        duration_of_pitch = st.number_input("Pitch duration (minutes)", min_value=0, max_value=150, value=15)
    with c4:
        num_followups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=4)
        pitch_score = st.slider("Pitch satisfaction score", 1, 5, 3)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "Age": float(age),
        "TypeofContact": type_of_contact,
        "CityTier": int(city_tier),
        "DurationOfPitch": float(duration_of_pitch),
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": int(num_person),
        "NumberOfFollowups": float(num_followups),
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": float(property_star),
        "MaritalStatus": marital_status,
        "NumberOfTrips": float(num_trips),
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": int(pitch_score),
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": float(num_children),
        "Designation": designation,
        "MonthlyIncome": float(monthly_income),
    }])

    proba = float(model.predict_proba(input_df)[0, 1])
    will_buy = proba >= threshold

    st.metric("Purchase probability", f"{proba:.1%}")
    if will_buy:
        st.success("Likely to buy. Prioritise this customer for outreach.")
    else:
        st.info("Unlikely to buy right now. Keep in a lower-priority nurture list.")

    with st.expander("Input sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "value"}).astype(str))

if meta:
    with st.sidebar:
        st.header("Model card")
        st.write(f"Version: **{meta.get('model_version', '?')}**")
        st.write(f"Trained: {meta.get('trained_at_utc', '?')}")
        m = meta.get("metrics", {})
        st.write(f"Test F1: {m.get('test_f1', 0):.3f}")
        st.write(f"Test recall: {m.get('test_recall', 0):.3f}")
        st.write(f"Test ROC-AUC: {m.get('test_roc_auc', 0):.3f}")
        st.caption(f"Decision threshold: {threshold}")
