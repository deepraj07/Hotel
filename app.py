"""
app.py
======
ABC Ltd — Hotel Booking Intelligence
A Streamlit decision-support tool for hotel managers.

Loads two pre-trained scikit-learn pipelines (preprocessing + model) and
produces, for a single booking entered by a manager:
  1. A cancellation probability (Logistic Regression)
  2. A predicted Average Daily Rate / ADR (Linear Regression)

This app does NOT load or require hotel_bookings.csv — only the two saved
.sav pipeline files are needed for deployment.
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# -----------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="ABC Ltd — Hotel Booking Intelligence",
    page_icon="🏨",
    layout="centered",
)

st.title("ABC Ltd — Hotel Booking Intelligence")
st.caption("AI-assisted booking cancellation and ADR prediction")

st.markdown(
    "This tool provides **decision support** for hotel managers evaluating "
    "a booking. It does **not** make the final decision — please use "
    "professional judgement alongside these estimates."
)

# -----------------------------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------------------------
CLF_PATH = "cancellation_logistic_reg.sav"
REG_PATH = "adr_linear_reg.sav"


@st.cache_resource
def load_models():
    clf_model = joblib.load(CLF_PATH)
    reg_model = joblib.load(REG_PATH)
    return clf_model, reg_model


missing_files = [f for f in (CLF_PATH, REG_PATH) if not os.path.exists(f)]
if missing_files:
    st.error(
        "The following required model file(s) could not be found: "
        f"{', '.join(missing_files)}. Please run `train_models.py` first, "
        "or make sure the .sav files were included in the deployment."
    )
    st.stop()

try:
    clf_pipeline, reg_pipeline = load_models()
except Exception as e:
    st.error(f"Could not load the saved models. Details: {e}")
    st.stop()

# -----------------------------------------------------------------------
# SIDEBAR — BOOKING INPUTS
# -----------------------------------------------------------------------
st.header("Booking details")

col1, col2 = st.columns(2)

with col1:
    hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
    lead_time = st.number_input(
        "Lead time (days between booking and arrival)",
        min_value=0, max_value=800, value=60, step=1,
    )
    arrival_date_year = st.selectbox("Arrival year", [2015, 2016, 2017], index=2)
    arrival_date_month = st.selectbox(
        "Arrival month",
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"],
        index=6,
    )
    arrival_date_week_number = st.number_input(
        "Arrival week number", min_value=1, max_value=53, value=27, step=1
    )
    arrival_date_day_of_month = st.number_input(
        "Arrival day of month", min_value=1, max_value=31, value=15, step=1
    )
    stays_in_weekend_nights = st.number_input(
        "Weekend nights", min_value=0, max_value=20, value=1, step=1
    )
    stays_in_week_nights = st.number_input(
        "Week nights", min_value=0, max_value=40, value=2, step=1
    )
    adults = st.number_input("Adults", min_value=0, max_value=10, value=2, step=1)
    children = st.number_input("Children", min_value=0, max_value=10, value=0, step=1)
    babies = st.number_input("Babies", min_value=0, max_value=10, value=0, step=1)

with col2:
    is_repeated_guest = st.selectbox("Repeated guest?", ["No", "Yes"])
    previous_cancellations = st.number_input(
        "Previous cancellations by this guest", min_value=0, max_value=30, value=0, step=1
    )
    previous_bookings_not_canceled = st.number_input(
        "Previous bookings NOT cancelled by this guest",
        min_value=0, max_value=80, value=0, step=1,
    )
    days_in_waiting_list = st.number_input(
        "Days on the waiting list", min_value=0, max_value=400, value=0, step=1
    )
    required_car_parking_spaces = st.number_input(
        "Required car parking spaces", min_value=0, max_value=8, value=0, step=1
    )
    total_of_special_requests = st.number_input(
        "Total special requests", min_value=0, max_value=10, value=0, step=1
    )
    meal = st.selectbox("Meal plan", ["BB", "HB", "FB", "SC", "Undefined"])
    market_segment = st.selectbox(
        "Market segment",
        ["Direct", "Corporate", "Online TA", "Offline TA/TO",
         "Complementary", "Groups", "Undefined", "Aviation"],
        index=2,
    )
    distribution_channel = st.selectbox(
        "Distribution channel", ["Direct", "Corporate", "TA/TO", "Undefined", "GDS"],
        index=2,
    )
    reserved_room_type = st.selectbox(
        "Reserved room type", ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
    )
    deposit_type = st.selectbox(
        "Deposit type", ["No Deposit", "Refundable", "Non Refund"]
    )
    customer_type = st.selectbox(
        "Customer type", ["Transient", "Transient-Party", "Contract", "Group"]
    )

# -----------------------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------------------
st.markdown("---")

if st.button("Predict Booking", type="primary"):
    input_row = pd.DataFrame([{
        "lead_time": lead_time,
        "arrival_date_year": arrival_date_year,
        "arrival_date_week_number": arrival_date_week_number,
        "arrival_date_day_of_month": arrival_date_day_of_month,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "is_repeated_guest": 1 if is_repeated_guest == "Yes" else 0,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "days_in_waiting_list": days_in_waiting_list,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,
        "hotel": hotel,
        "arrival_date_month": arrival_date_month,
        "meal": meal,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "reserved_room_type": reserved_room_type,
        "deposit_type": deposit_type,
        "customer_type": customer_type,
    }])

    try:
        cancel_proba = clf_pipeline.predict_proba(input_row)[0, 1]
        predicted_adr = reg_pipeline.predict(input_row)[0]
    except Exception as e:
        st.error(
            "Something went wrong while generating the prediction. This can "
            "happen if an input value is outside what the model expects. "
            f"Details: {e}"
        )
        st.stop()

    predicted_adr = max(predicted_adr, 0)  # ADR cannot sensibly be negative

    st.header("Prediction results")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Cancellation risk", f"{cancel_proba * 100:.0f}%")
    with res_col2:
        st.metric("Expected ADR", f"€{predicted_adr:.0f}/day")

    # Managerial interpretation — cancellation
    if cancel_proba >= 0.66:
        st.warning(
            "**High cancellation risk** — consider reviewing the booking "
            "and appropriate revenue/retention actions."
        )
    elif cancel_proba >= 0.33:
        st.info("**Moderate cancellation risk** — monitor the booking.")
    else:
        st.success("**Low cancellation risk** based on model prediction.")

    st.caption(
        "Predicted ADR represents the estimated average daily room rate "
        "based on the booking characteristics."
    )

    st.markdown("---")
    st.subheader("Why might the model be making this prediction?")
    st.caption(
        "**Important model factors** (general, model-level patterns from "
        "training — not an individual, booking-specific explanation)."
    )

    st.markdown(
        "- Bookings with **no deposit / refundable deposits**, **longer lead "
        "times**, and certain **online/undefined market segments** are "
        "generally associated with **higher** cancellation risk in the "
        "training data.\n"
        "- Bookings with **required parking spaces**, **more special "
        "requests**, and **non-refundable-friendly deposit patterns** tend "
        "to be associated with **lower** cancellation risk.\n"
        "- ADR tends to be **higher** in peak months (e.g. summer) and for "
        "certain room types and market segments, and **lower** in low "
        "season months (e.g. January–March) and complementary bookings."
    )
    st.caption(
        "These are general associations learned across many historical "
        "bookings, not a causal or individualized explanation of this "
        "specific prediction."
    )

st.markdown("---")
st.caption(
    "Predictions are decision-support estimates produced by statistical "
    "models trained on historical data. They should not be treated as "
    "deterministic decisions, and do not replace managerial judgement."
)
