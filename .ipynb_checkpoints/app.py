import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the trained model and preprocessing pipeline
model = joblib.load('best_model.pkl')
pipeline = joblib.load('pipeline.pkl')

# Function to get the user input
def get_user_input():
    loan_amnt = st.sidebar.slider('Loan Amount', 500, 50000, 5000)
    term = st.sidebar.selectbox('Term', (0, 1))
    int_rate = st.sidebar.slider('Interest Rate', 5.0, 31.0, 15.0)
    installment = st.sidebar.slider('Installment', 4.93, 1719.83, 100.0)
    fico_average = st.sidebar.slider('FICO Score', 0, 1000, 500)
    annual_inc = st.sidebar.slider('Annual Income', 0, 10999200, 50000)
    dti = st.sidebar.slider('Debt-to-Income Ratio', 0, 41, 20)
    open_acc = st.sidebar.slider('Number of Open Accounts', 0, 90, 10)
    revol_bal = st.sidebar.slider('Revolving Balance', 0, 2904836, 10000)
    revol_util = st.sidebar.slider('Revolving Line Utilization', 0.0, 892.30, 50.0)
    total_acc = st.sidebar.slider('Total Number of Credit Lines', 2, 176, 10)
    mort_acc = st.sidebar.selectbox('Mortgage Accounts', (0, 1))
    pub_rec_bankruptcies = st.sidebar.selectbox('Public Record Bankruptcies', (0, 1))
    year = st.sidebar.slider('Year', 1934, 2100, 2000)
    emp_length = st.sidebar.slider('Employment Length (years)', 0, 10, 5)
    home_ownership = st.sidebar.selectbox('Home Ownership', ('RENT', 'OWN', 'MORTGAGE', 'OTHER'))
    initial_list_status = st.sidebar.selectbox('Initial List Status', ('W', 'F'))
    verification_status = st.sidebar.selectbox('Verification Status', ('Source Verified', 'Verified', 'Not Verified'))
    purpose = st.sidebar.selectbox('Purpose', ('debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase', 'medical', 'small_business', 'car', 'moving', 'vacation', 'house', 'wedding', 'renewable_energy', 'educational'))
    region = st.sidebar.selectbox('Region', ('west', 'south_west', 'south_east', 'mid_west', 'north_east'))
    application_type = st.sidebar.selectbox('Application Type', ('Individual', 'Joint App'))

    # Store a dictionary into a dataframe
    user_data = {
        'loan_amnt': loan_amnt,
        'term': term,
        'int_rate': int_rate,
        'installment': installment,
        'fico_average': fico_average,
        'annual_inc': annual_inc,
        'dti': dti,
        'open_acc': open_acc,
        'revol_bal': revol_bal,
        'revol_util': revol_util,
        'total_acc': total_acc,
        'mort_acc': mort_acc,
        'pub_rec_bankruptcies': pub_rec_bankruptcies,
        'year': year,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'initial_list_status': initial_list_status,
        'verification_status': verification_status,
        'purpose': purpose,
        'region': region,
        'application_type': application_type,
    }

    features = pd.DataFrame(user_data, index=[0])
    return features

# Get user input
user_input = get_user_input()

st.write('# Loan Eligibility App')

st.write('## User Input:')
st.write(user_input)

# Apply preprocessing and make a prediction
preprocessed_input = pipeline.transform(user_input)
risk_score = model.predict_proba(preprocessed_input)
risk_class = 'High Risk' if risk_score[0][1] > 0.5 else 'Low Risk'

st.write('## Risk Classification:')
st.write(risk_class)

st.write('## Probability of Default:')
st.write(risk_score[0][1])