import streamlit as st
import pandas as pd, numpy as np
from utils import (
    load_data, describe_columns, plot_explore_charts,
    train_and_evaluate_models, predict_and_download_df
)
st.set_page_config(page_title="Universal Bank - Personal Loan ML", layout="wide")

st.title("Universal Bank — Customer Personal Loan Prediction")
st.markdown("A Streamlit dashboard to explore the Universal Bank data, train three tree models, and predict Personal Loan acceptance.")

# Sidebar - data selection / upload
st.sidebar.header("Data")
use_sample = st.sidebar.checkbox("Use sample dataset bundled with app", value=True)
uploaded = st.sidebar.file_uploader("Or upload your UniversalBank CSV (no folders in zip).", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = pd.read_csv("UniversalBank_sample.csv")
else:
    st.warning("Please upload a CSV or enable the sample dataset on the sidebar.")
    st.stop()

# Normalize column names (allow variations)
df.columns = [c.strip() for c in df.columns]

# Show column description table
if st.sidebar.checkbox("Show data dictionary", value=True):
    desc_df = describe_columns(df.columns)
    st.sidebar.dataframe(desc_df, height=400)

tabs = st.tabs(["Explore", "Modeling", "Predict"])

# ------------- Explore tab -------------
with tabs[0]:
    st.header("Exploratory charts & marketing insights")
    st.markdown("Interactive charts to help identify target segments and actions for better conversion rates.")
    plot_explore_charts(df, container=st)

# ------------- Modeling tab -------------
with tabs[1]:
    st.header("Train & Evaluate Models (Decision Tree / Random Forest / Gradient Boosting)")
    st.markdown("Click **Train models** to run the full pipeline (train/test split, 5-fold CV, metrics, ROC, confusion matrices, feature importances).")
    if st.button("Train models"):
        with st.spinner("Training models and computing metrics..."):
            results = train_and_evaluate_models(df, container=st)
            # results contains metrics_df and cv_df if needed
            st.success("Training complete. Scroll to see metrics and plots.")

# ------------- Predict tab -------------
with tabs[2]:
    st.header("Upload new data and predict Personal Loan label")
    st.markdown("Upload a CSV with the same columns (except `Personal Loan` allowed to be missing). The app will return a CSV with predicted `Personal Loan` probabilities and labels.")
    newfile = st.file_uploader("Upload new customer CSV for prediction", type=["csv"], key="predict_upload")
    if newfile is not None:
        newdf = pd.read_csv(newfile)
        st.write("Preview of uploaded data:")
        st.dataframe(newdf.head())
        if st.button("Predict on uploaded file"):
            with st.spinner("Running prediction..."):
                out_df = predict_and_download_df(newdf, reference_df=pd.read_csv("UniversalBank_sample.csv"))
                st.success("Predictions ready. Download below.")
                st.download_button("Download predictions CSV", out_df.to_csv(index=False).encode('utf-8'), file_name="predictions_universalbank.csv", mime="text/csv")
    else:
        st.info("Upload a CSV to enable batch predictions. If you want to test, enable 'Use sample dataset' and go to Modeling to train models first.")

st.markdown("---")
st.caption("Built for easy deployment on Streamlit Cloud. Keep all files at repository root (no folders).")
