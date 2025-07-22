import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import os
import pickle

# App title and header
st.write("""
# Concrete Compressive Strength Prediction
This app predicts the **Concrete Compressive Strength** of eco-friendly concrete containing **RA and GGBFS** using Machine Learning!
""")
st.write('---')

# Load and display image
try:
    image = Image.open('Concrete-compressive-strength-test.png')
    st.image(image, use_column_width=True)
except FileNotFoundError:
    st.warning("Image file not found.")

# Load dataset
try:
    data = pd.read_csv("paperbinder.csv")
except FileNotFoundError:
    st.error("The file 'paperbinder.csv' was not found.")
    st.stop()

# Rename columns
req_col_names = ["Wateroverbinder", "Recycled_aggregate_percentage", "GGBFS_percentage_", "Superplasticizer", "Age", "CC_Strength (MPa)"]
curr_col_names = list(data.columns)
mapper = {curr_col_names[i]: req_col_names[i] for i in range(len(curr_col_names))}
data = data.rename(columns=mapper)

# Show dataset info
st.subheader('Data Information')
st.dataframe(data)

# Check for missing values
if data.isna().sum().sum() > 0:
    st.warning("Dataset contains missing values. Please handle them before training.")
else:
    st.success("No missing values detected.")

# Correlation (not displayed as heatmap here)
corr = data.corr()

# Feature and target split
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Sidebar inputs
st.sidebar.header('Specify Input Parameters')
def get_input_features():
    Wateroverbinder = st.sidebar.slider('W/B', 0.25, 0.75, 0.3)
    Recycled_aggregate_percentage = st.sidebar.slider('RA%', 0, 100, 50)
    GGBFS_percentage_ = st.sidebar.slider('GGBFS%', 0, 90, 15)
    Superplasticizer = st.sidebar.slider('Superplasticizer (kg)', 0.0, 7.8, 2.0)
    Age = st.sidebar.slider('Age (days)', 7, 90, 8)
    data_user = {
        'Wateroverbinder': Wateroverbinder,
        'Recycled_aggregate_percentage': Recycled_aggregate_percentage,
        'GGBFS_percentage_': GGBFS_percentage_,
        'Superplasticizer': Superplasticizer,
        'Age': Age
    }
    return pd.DataFrame(data_user, index=[0])

df = get_input_features()

# Show input parameters
st.header('Specified Input Parameters')
st.write(df)
st.write('---')

# Check current working directory and list files (debug step)
st.write("Current working directory:", os.getcwd())
st.write("Files in current directory:", os.listdir())

# Load trained model
model_path = 'concrete_eco.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        try:
            load_clf = pickle.load(file)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()
else:
    st.error(f"Model file '{model_path}' not found in the current directory.")
    st.stop()

# Make prediction
st.header('Prediction of Concrete Compressive Strength (MPa)')
try:
    prediction = load_clf.predict(df)
    st.success(f"Predicted Strength: {prediction[0]:.2f} MPa")
except Exception as e:
    st.error(f"Prediction failed: {e}")

st.write('---')
