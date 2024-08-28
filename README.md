### Automated Machine Learning Web app using PyCaret and Pandas.

This repository contains an Automated Machine Learning (AutoML) web application designed to simplify the process of building and evaluating machine learning models. 
The app allows users to upload datasets, perform automated machine learning, and download the best model for deployment.

## Features

- **Upload Datasets**: Users can upload CSV files for analysis.
- **Automated Model Selection**: The app automatically selects the best-performing model from various algorithms.
- **Model Evaluation**: Detailed performance metrics are provided for each model.
- **Download Best Model**: Users can download the best-performing model as a `.pkl` file for future use.
- **Logs**: The app maintains logs of operations for debugging and auditing purposes.

  The main libraries and frameworks used in this project are:
  
- Scikit-learn
- Pandas
- PyCaret
- Joblib
- Numba
- Pandas-Profiling
- Sktime
- Streamlit

For the full list of dependencies, please refer to the `requirements.txt` file.

## TO RUN: streamlit run app.py
