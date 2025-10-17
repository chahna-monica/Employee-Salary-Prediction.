import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("xgboost_smote.pkl")  
preprocessor = joblib.load("preprocessor.pkl")

st.set_page_config(page_title="Income Prediction", layout="wide")
st.title("Income Level Prediction (<=50K or >50K)")

def user_input_form():
    age = st.number_input("Age", min_value=17, max_value=90, value=30)
    workclass = st.selectbox("Workclass", [
        'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
        'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
    ])
    fnlwgt = st.number_input("Final Weight (fnlwgt)", value=100000)
    education = st.selectbox("Education", [
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school'
    ])
    educational_num = st.slider("Education Number", 1, 16, 10)
    marital_status = st.selectbox("Marital Status", [
        'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
        'Widowed', 'Married-spouse-absent'
    ])
    occupation = st.selectbox("Occupation", [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
        'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
        'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
        'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
    ])
    relationship = st.selectbox("Relationship", [
        'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'
    ])
    race = st.selectbox("Race", [
        'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'
    ])
    sex = st.selectbox("Sex", ['Male', 'Female'])
    gender = st.selectbox("Gender", ['Male', 'Female']) 
    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)
    hours_per_week = st.slider("Hours per Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", [
        'United-States', 'Mexico', 'Philippines', 'Germany', 'Canada',
        'India', 'England', 'China', 'Cuba', 'Iran', 'South', 'Other'
    ])

    data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'education': education,
        'educational-num': educational_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }

    return pd.DataFrame([data])

input_df = user_input_form()

if st.button("Predict Income"):
    try:
        input_processed = preprocessor.transform(input_df)
        prediction = model.predict(input_processed)[0]
        proba = model.predict_proba(input_processed)[0][1]

        label = ">50K" if prediction == 1 else "<=50K"
        st.success(f"Predicted Income: {label}")
        st.info(f"Confidence: {proba:.2f}")

    except Exception as e:
        st.error(f"Error in prediction: {e}")
