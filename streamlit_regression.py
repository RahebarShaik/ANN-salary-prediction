import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import pandas as pd
import pickle

#Load the train model
model = tf.keras.models.load_model('regression_model.h5')

with open("label_encoder_gender_reg.pkl",'rb') as f:
    label_encoder_gender = pickle.load(f) 

with open("one_hot_encoder_geo_reg.pkl", 'rb') as f:
    OHE_geo = pickle.load(f)

with open("scaler_reg.pkl", 'rb') as f:
    scaler = pickle.load(f)


## StreamLit app
st.title("Customer Salary Estimation")

#Input fields
geography = st.selectbox("Geography", OHE_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 100)
balance = st.number_input("Balance", min_value=0.0)
exited = st.selectbox("Exited",[0,1])
credit_score = st.number_input("Credit Score", min_value=0)
tenure = st.slider("Tenure", 0, 10)
no_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

#Prepare the input data
# Example input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [no_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# One-hot encode 'Geography'
geo_encoded = OHE_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=OHE_geo.get_feature_names_out(['Geography']))

# Combine all features into a single DataFrame
input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df],axis=1)

# Scale the input data
input_scaled = scaler.transform(input_df)

## Predict churn
prediction = model.predict(input_scaled)
prediction_salary = prediction[0][0]

# st.subheader(f"Churn Probability: {prediction_proba:.2%}")

st.subheader(f"Estimated Salary: ${prediction_salary:.2f}")