
import streamlit as st 
import pandas as pd
import os
import shap
import lime
import lime.lime_tabular

# Importing for Data Profiling
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

#Machine Learning part
from pycaret.classification import setup,compare_models, pull, save_model, load_model

with st.sidebar:
    st.image("https://i.pinimg.com/236x/73/cd/2a/73cd2a910f69d1fb3a61979364a22bff.jpg")
    st.title("ML Maestro")
    choice = st.radio("Navigation", ["Upload Dataset", "Data Profiling", "ML", "Download Model"])
    st.info("This is a Zero Code Web Application to build Automated ML pipeline using Streamlit, Pandas profiling and PyCaret.")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload Dataset":
    st.title("Upload Your Dataset For Modelling")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)        
        st.dataframe(df)

if choice == "Data Profiling":
    st.title("Automated Exploratory Data Analysis")
    if 'df' in locals():
        profile_report = ProfileReport(df)
        st_profile_report(profile_report)
    else:
        st.error("No data uploaded yet. Please upload a dataset.")

if choice == "ML":
    st.title("Running for ML Model")
    target = st.selectbox("Select Your Target", df.columns)
    if st.button("Train model"):
        setup(df,target=target)
        setup_df = pull()
        st.info("This is the ML experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model,'best_model')

if choice == "Download Model":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the model", f, "trained_model.pkl")
