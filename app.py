import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib
from functions import Outlier_Drop_and_Skewness_handler, features_to_drop, one_hot_encoding, FeatureHashing, OrdinalFeatNames, MinMaxWithFeatNames, Oversample


# Function to load the trained model and preprocessing pipeline
def load_model_pipeline():
    try:
        with open('best_model.pkl', 'rb') as f:
            model = joblib.load(f)
        with open('pipeline.pkl', 'rb') as f:
            pipeline = joblib.load(f)
        return model, pipeline
    except Exception as e:
        st.error(f"Error loading model or pipeline: {e}")
        return None, None


# Function to get the user input
def get_user_input():
    st.sidebar.markdown("## 💼 User Parameters")
    loan_amnt = st.sidebar.slider('Loan Amount', 500, 50000, 5000)
    term = st.sidebar.selectbox('Term', (0, 1))
    sub_grade = st.sidebar.selectbox('Loan sub grades', ('A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5','D1','D2','D3','D4','D5','E1','E2','E3','E4','E5','F1','F2','F3','F4','F5','G1','G2','G3','G4','G5'))
    int_rate = st.sidebar.slider('Interest Rate', 5.0, 31.0, 15.0)
    installment = st.sidebar.slider('Installment', 4.93, 1719.83, 100.0)
    Fico_average = st.sidebar.slider('FICO Score', 0, 1000, 500)
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
        'sub_grade': sub_grade,
        'emp_length': emp_length,
        'annual_inc': annual_inc,
        'dti': dti,
        'open_acc': open_acc,
        'revol_bal': revol_bal,
        'revol_util': revol_util,
        'total_acc': total_acc,
        'mort_acc': mort_acc,
        'pub_rec_bankruptcies': pub_rec_bankruptcies,
        'Fico_average': Fico_average,
        'year': year,
        'home_ownership': home_ownership,
        'verification_status': verification_status,
        'purpose': purpose,
        'initial_list_status': initial_list_status,
        'application_type': application_type,
        'region': region,
    }

    features = pd.DataFrame(user_data, index=[0])
    return features

def make_prediction(user_input, model, pipeline):
    """Apply preprocessing and make a prediction."""
    try:
        preprocessed_input = pipeline[1:-1].transform(user_input)
        risk_score = model.predict_proba(preprocessed_input)
        return risk_score
    except Exception as e:
        st.error(f"🔴 Error occurred while making the prediction: {e}")
        return None
    
def main():

    # Load the trained model and preprocessing pipeline
    model, pipeline = load_model_pipeline()

    # Return if model or pipeline failed to load
    if model is None or pipeline is None:
        return

    st.set_page_config(page_title="CreditRisk Predictor", layout="wide")
    
    # Get user input
    user_input = get_user_input()

    st.markdown('# 🏦 CreditRisk Predictor')

    # App description
    st.markdown("## App Description")
    st.markdown('The CreditRisk Predictor App is a web application that predicts the risk of loan default based on user input. Users provide their personal and financial details, and the app uses a pre-trained machine learning model to calculate the risk score. The user can set a risk threshold to determine the risk classification as "High Risk" or "Low Risk." The app helps users make informed decisions by assessing the probability of loan default and providing a quick risk assessment for loan applications.')
        
    st.markdown('## 👤 User Input Parameters:')
    st.table(user_input)  # use table for cleaner formatting

    # Get risk threshold from user
    st.markdown("## ⚠️ Set Your Risk Threshold")
    risk_threshold = st.slider("Choose your risk tolerance", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    if st.button('Predict Risk'):
        # Make prediction and handle exceptions
        risk_score = make_prediction(user_input, model, pipeline)

        # Display the risk classification and probability of default if prediction was successful
        if risk_score is not None:
            risk_class = 'High Risk' if risk_score[0][1] > risk_threshold else 'Low Risk'

            st.markdown('## 🎯 Risk Classification:')
            st.success(risk_class)

            st.markdown('## 📈 Probability of Default:')
            st.info(f'{risk_score[0][1]:.2%}')  # Format the probability as a percentage with 2 decimal points

if __name__ == '__main__':
    main()