import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json

# ----------------------------------------
# Load model, columns, and processed data
# ----------------------------------------
@st.cache_data
def load_resources():
    with open("banglore_home_prices_model.pickle", "rb") as f:
        model = pickle.load(f)

    with open("columns.json", "r") as f:
        cols = json.load(f)["data_columns"]

    df = pd.read_csv("hpp_excet.csv")
    df["location"] = df["location"].str.strip()
    df["category"] = df["category"].str.lower().str.strip()

    return model, cols, df


model, data_columns, df = load_resources()


# ----------------------------------------
# Prediction Logic (same as training code)
# ----------------------------------------
def predict_price(location, sqft, bhk, area_type):
    area_type = area_type.lower().strip()
    X_columns = data_columns

    # Step 1 — find actual type of this location
    sample = df[df["location"].str.lower() == location.lower()]
    if len(sample) == 0:
        return "❌ Location not found in training data!"

    actual_category = sample["category"].iloc[0]  # urban / rural

    # Step 2 — validate user's area type
    if area_type != actual_category:
        return f"❌ This location is **{actual_category.title()}**, not {area_type.title()}, Please Select {area_type.title()} location."

    # Step 3 — Build input vector
    x = np.zeros(len(X_columns))

    # numeric features
    x[X_columns.index("total_sqft")] = sqft
    x[X_columns.index("bhk")] = bhk

    # location one-hot
    loc = location.lower().strip()
    if loc in X_columns:
        x[X_columns.index(loc)] = 1

    # area type one-hot
    area_col = f"area_{area_type}"
    if area_col in X_columns:
        x[X_columns.index(area_col)] = 1

    # Step 4 — predict
    return round(model.predict([x])[0], 2)


# ----------------------------------------
# Streamlit UI
# ----------------------------------------
st.set_page_config(page_title="🏡 House Price Prediction",
                   page_icon="🏠",
                   layout="centered")

st.title("🏡 House Price Prediction")


# ----------------------------------------
# User Inputs
# ----------------------------------------
st.header("Enter Property Details")

sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, step=10)

bhk = st.selectbox("BHK", [1,2,3,4,5,6,7,8,9,10])

locations = sorted(df["location"].unique())
location = st.selectbox("Select Location", locations)

area_type = st.radio("Area Type", ["Urban", "Rural"])

predict_btn = st.button("Predict Price 🔍")

# ----------------------------------------
# Output Page
# ----------------------------------------
if predict_btn:
    result = predict_price(location, sqft, bhk, area_type)

    st.markdown("## 🏁 Prediction Result")

    if isinstance(result, str) and result.startswith("❌"):
        st.error(result)
    else:
        st.success(f"### 💰 Estimated Price: **₹ {result} Lakhs**")

        st.write("---")
        st.info(f"""
        **Summary**  
        • Location: {location}  
        • Sqft: {sqft}  
        • BHK: {bhk}  
        • Area Type: {area_type}  
        """)

st.markdown("---")

