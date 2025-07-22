import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import numpy as np
import pandas as pd
import csv
import streamlit as st
from PIL import Image

st.write("""
# Concrete Compressive Strength Prediction
This app predicts the **Concrete Compressive Strength of eco-friednly containing RA and GGBFS using Machine Learning**!
""")
st.write('---')
image=Image.open(r'Concrete-compressive-strength-test.png')
st.image(image, use_column_width=True)

data = pd.read_csv(r"paperbinder.csv")

req_col_names = ["Wateroverbinder", "Recycled_aggregate_percentage", "GGBFS_percentage_", "Superplasticizer", "Age", "CC_Strength (MPa)"]
curr_col_names = list(data.columns)

mapper = {}
for i, name in enumerate(curr_col_names):
    mapper[name] = req_col_names[i]

data = data.rename(columns=mapper)
st.subheader('data information')
data.head()
data.isna().sum()
corr = data.corr()
st.dataframe(data)

X = data.iloc[:,:-1]         # Features - All columns but last
y = data.iloc[:,-1]          # Target - Last Column
print(X)
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
st.sidebar.header('Specify Input Parameters')

def get_input_features():
    Wateroverbinder = st.sidebar.slider('W/B', 0.25,0.75,0.3)
    Recycled_aggregate_percentage = st.sidebar.slider('RA%',0,100,50)
    GGBFS_percentage_ = st.sidebar.slider('GGBFS%', 0,90,15)
    Superplasticizer = st.sidebar.slider('Superplasticizer (kg)', 0.0,7.8,2.0)
    Age = st.sidebar.slider('Age (days)', 7,90,8)

    data_user = {'Wateroverbinder': Wateroverbinder,
            'Recycled_aggregate_percentage': Recycled_aggregate_percentage,
            'GGBFS_percentage_': GGBFS_percentage_,
            'Superplasticizer': Superplasticizer,
            'Age': Age}
    features = pd.DataFrame(data_user, index=[0])
    return features

df = get_input_features()
# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')




# Reads in saved classification model
import pickle
load_clf = pickle.load(open('concrete_eco.pkl', 'rb'))
st.header('Prediction of Concrete Compressive Strength (Mpa)')

# Apply model to make predictions
prediction = load_clf.predict(df)
st.write(prediction)
st.write('---')
