import tensorflow as tf
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pickle

# Load the trained model
model=tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')

# User input
geograpgy = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score= st.number_input('CreditScore')
estimated_scalary=st.number_input('EstimatedSalary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('NumOfProducts',1,4)
has_cr_card= st.selectbox('HasCrCard',[0,1])
is_active_member=st.selectbox('IsActiveMember',[0,1])

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_scalary],
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geograpgy]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Concatenate One-hot encoded column with input data
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled=scaler.transform(input_data)

# Predict churn
prediction=model.predict(input_data_scaled)
prediction_prob=prediction[0][0]

st.write(f'Churn Probability : {prediction_prob:.2f}')

if prediction_prob > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is unlikely to churn.')


