import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Load the trained model
model = load_model('model.keras')

# Load the encoders and scaler
with open('gender_le.pkl', 'rb') as file: 
    gender_le = pickle.load(file)

with open('geography_ohe.pkl', 'rb') as file: 
    geography_ohe = pickle.load(file)

with open('scaler.pkl', 'rb') as file: 
    scaler = pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')

# User Input
geography = st.selectbox('Geography', geography_ohe.categories_[0])
gender = st.selectbox('Gender', gender_le.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=0, max_value=1000)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_le.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = geography_ohe.transform(pd.DataFrame({'Geography': [geography]}))
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geography_ohe.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0] * 100

# print probability
st.write(f'Churn Probability: {prediction_prob:.2f}%')


if prediction > 0.5: 
    st.write('The customer is likely to churn')
else: 
    st.write('The customer is not likely to churn')


