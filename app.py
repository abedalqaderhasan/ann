import streamlit as st
import numpy as np
import tensorflow as tf  # Ensure TensorFlow is installed in your environment
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

# Load encoders with error handling
try:
    with open('label_encoder_gender.pkl' , 'rb') as file : 
        label_encoder_gender = pickle.load(file)
    # Test if the encoder is properly fitted
    label_encoder_gender.transform(['Male'])
except:
    # Fallback: create a new label encoder for gender
    label_encoder_gender = LabelEncoder()
    label_encoder_gender.fit(['Male', 'Female'])

try:
    with open('onehot_encoder_geo.pkl' , 'rb') as file : 
        onehot_encoder_geo = pickle.load(file)
    # Test if the encoder is properly fitted
    onehot_encoder_geo.transform([['France']])
except:
    # Fallback: create a new one-hot encoder for geography
    onehot_encoder_geo = OneHotEncoder(sparse_output=False, drop=None)
    onehot_encoder_geo.fit([['France'], ['Germany'], ['Spain']])

try:
    with open('scaler.pkl' , 'rb') as file : 
        scaler = pickle.load(file)
except:
    # Fallback: create a new scaler
    scaler = StandardScaler()
    st.warning("⚠️ Scaler file not found. Predictions may be inaccurate.")

st.title('Customer Churn Prediction')

# Input fields
st.header('Enter Customer Information:')

# Required customer information fields
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600, step=1, 
                              help='Customer credit score (300-850)')

geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'], 
                        help='Customer location')

gender = st.selectbox('Gender', ['Male', 'Female'], 
                     help='Customer gender')

age = st.number_input('Age', min_value=18, max_value=100, value=35, step=1,
                     help='Customer age')

tenure = st.number_input('Tenure (Years)', min_value=0, max_value=20, value=5, step=1,
                        help='Number of years as customer')

balance = st.number_input('Account Balance', min_value=0.0, value=50000.0, step=100.0,
                         help='Customer account balance')

num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=2, step=1,
                                 help='Number of bank products customer has')

has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'],
                          help='Does customer have a credit card?')

is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'],
                               help='Is customer an active member?')

estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0, step=1000.0,
                                  help='Customer estimated annual salary')

# Predict button
if st.button('Predict Churn', type='primary'):
    try:
        # Prepare the input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Geography': [geography],
            'Gender': [gender],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
            'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
            'EstimatedSalary': [estimated_salary]
        })
        
        # Apply preprocessing
        # Handle Gender encoding
        input_data['Gender'] = label_encoder_gender.transform([gender])[0]
        
        # Handle Geography encoding
        geo_encoded = onehot_encoder_geo.transform([[geography]])
        
        # Create feature names that match training data
        geo_feature_names = ['Geography_France', 'Geography_Germany', 'Geography_Spain']
        # If using drop='first', remove the first category
        if hasattr(onehot_encoder_geo, 'drop') and onehot_encoder_geo.drop == 'first':
            geo_feature_names = geo_feature_names[1:]  # Remove Geography_France
        elif hasattr(onehot_encoder_geo, 'drop') and onehot_encoder_geo.drop is not None:
            # Handle other drop cases if needed
            try:
                feature_names = onehot_encoder_geo.get_feature_names_out(['Geography'])
                geo_feature_names = list(feature_names)
            except:
                geo_feature_names = ['Geography_Germany', 'Geography_Spain']
        
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_feature_names)
        # Combine the data
        input_data = input_data.drop('Geography', axis=1)
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        
        # Scale the data
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_scaled)
        prediction_prob = prediction[0][0]

        print(prediction_prob)
        
        # Display results
        st.header('Prediction Results:')
        
        if prediction_prob > 0.5:
            st.error(f'⚠️ High Risk of Churn: {prediction_prob:.2%} probability')
            st.write('**Recommendation:** Take immediate action to retain this customer.')
        else:
            st.success(f'✅ Low Risk of Churn: {prediction_prob:.2%} probability') 
            st.write('**Recommendation:** Customer likely to stay, maintain current service level.')
            
        # Additional insights
        st.subheader('Risk Analysis:')
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Churn Probability", f"{prediction_prob:.2%}")
        with col2:
            risk_level = "High" if prediction_prob > 0.5 else "Low"
            st.metric("Risk Level", risk_level)
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Please check that all required model files are present:")
        st.write("- model.h5")
        st.write("- label_encoder_gender.pkl") 
        st.write("- onehot_encoder_geo.pkl")
        st.write("- scaler.pkl")

# Footer
st.markdown("---")
st.markdown("*Customer Churn Prediction Model - Built with Streamlit and TensorFlow*")