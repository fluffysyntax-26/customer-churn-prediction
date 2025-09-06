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
st.markdown("Enter the customer details below to predict the likelihood of churn.")

# User Input - organized into columns for a cleaner look
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
    geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.slider('Age', 18, 92, 38)
    tenure = st.slider('Tenure (Years)', 0, 10, 5)

with col2:
    balance = st.number_input('Balance', value=60000.0, format="%.2f")
    num_of_products = st.slider('Number of Products', 1, 4, 1)
    # Use "Yes"/"No" for a better user experience
    has_cr_card_str = st.selectbox('Has Credit Card?', ['Yes', 'No'])
    is_active_member_str = st.selectbox('Is Active Member?', ['Yes', 'No'])
    estimated_salary = st.number_input('Estimated Salary', value=50000.0, format="%.2f")

# --- Prediction Logic ---
if st.button('Predict Churn'):
    # Convert user-friendly "Yes"/"No" back to 1/0 for the model
    has_cr_card = 1 if has_cr_card_str == 'Yes' else 0
    is_active_member = 1 if is_active_member_str == 'Yes' else 0

    # Create a dictionary with all the user inputs
    input_data_dict = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }

    # Create a DataFrame from the dictionary
    input_df = pd.DataFrame([input_data_dict])

    # --- Preprocessing ---
    # Encode Gender
    input_df['Gender'] = input_df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

    # One-hot encode 'Geography' using the loaded encoder object
    geo_encoded_array = geography_ohe.transform(input_df[['Geography']])
    geo_encoded_df = pd.DataFrame(geo_encoded_array, columns=geography_ohe.get_feature_names_out(['Geography']))
    
    # Drop the original 'Geography' column and concatenate the new encoded columns
    input_df = input_df.drop('Geography', axis=1)
    input_df = pd.concat([input_df, geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_df)

    # --- Prediction and Display ---
    prediction_prob = model.predict(input_data_scaled)
    churn_probability = prediction_prob[0][0]

    st.subheader('Prediction Result')
    st.write(f'**Churn Probability:** {churn_probability:.2%}')

    if churn_probability > 0.5:
        st.error('This customer is LIKELY to churn.')
    else:
        st.success('This customer is LIKELY to stay.')
