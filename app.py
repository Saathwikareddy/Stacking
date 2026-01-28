# Smart Loan Approval System – Stacking Model
# Streamlit Application

import streamlit as st
import numpy as np

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(page_title="Smart Loan Approval System", layout="wide")

# --------------------------------------------------
# App Title & Description
# --------------------------------------------------
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.markdown(
    "This system uses a **Stacking Ensemble Machine Learning model** to predict whether a loan will be approved by \n"
    "combining multiple ML models for better decision making."
)

# --------------------------------------------------
# Sidebar – Input Section
# --------------------------------------------------
st.sidebar.header("📝 Applicant Details")

applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapplicant_income = st.sidebar.number_input("Co-Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Amount Term (months)", min_value=0)

credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])

# --------------------------------------------------
# Model Architecture Display
# --------------------------------------------------
st.subheader("🧩 Stacking Model Architecture")
st.markdown(
    "**Base Models Used:**\n"
    "- Logistic Regression\n"
    "- Decision Tree\n"
    "- Random Forest\n\n"
    "**Meta Model Used:**\n"
    "- Logistic Regression"
)

# --------------------------------------------------
# Dummy Model Predictions (Simulation)
# --------------------------------------------------
# NOTE: Replace these with real trained models in production

def logistic_regression_model(features):
    return 1 if features[0] > 5000 and features[4] == 1 else 0


def decision_tree_model(features):
    return 1 if features[2] < 300 and features[4] == 1 else 0


def random_forest_model(features):
    return 1 if (features[0] + features[1]) > 7000 else 0


def meta_model(base_preds):
    return 1 if sum(base_preds) >= 2 else 0

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    # Encoding categorical inputs
    credit = 1 if credit_history == "Yes" else 0
    employment_val = 1 if employment == "Salaried" else 0
    property_val = {"Urban": 2, "Semi-Urban": 1, "Rural": 0}[property_area]

    features = np.array([
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit,
        employment_val,
        property_val
    ])

    # Base model predictions
    lr_pred = logistic_regression_model(features)
    dt_pred = decision_tree_model(features)
    rf_pred = random_forest_model(features)

    base_predictions = [lr_pred, dt_pred, rf_pred]

    # Meta-model prediction
    final_pred = meta_model(base_predictions)

    # --------------------------------------------------
    # Output Section
    # --------------------------------------------------
    st.subheader("📊 Prediction Results")

    if final_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.markdown("### 📊 Base Model Predictions")
    st.write(f"Logistic Regression → {'Approved' if lr_pred else 'Rejected'}")
    st.write(f"Decision Tree → {'Approved' if dt_pred else 'Rejected'}")
    st.write(f"Random Forest → {'Approved' if rf_pred else 'Rejected'}")

    confidence = int((sum(base_predictions) / 3) * 100)

    st.markdown("### 🧠 Final Stacking Decision")
    st.write(f"📈 Confidence Score: **{confidence}%**")

    # --------------------------------------------------
    # Business Explanation
    # --------------------------------------------------
    st.subheader("💼 Business Explanation")

    if final_pred == 1:
        st.markdown(
            "Based on the applicant's income, credit history, and the combined predictions from multiple machine "
            "learning models, the applicant is **likely to repay the loan**. \n\n"
            "Therefore, the **stacking model predicts loan approval**."
        )
    else:
        st.markdown(
            "Based on the applicant's income, credit history, and the combined predictions from multiple machine "
            "learning models, the applicant is **unlikely to repay the loan**. \n\n"
            "Therefore, the **stacking model predicts loan rejection**."
        )
