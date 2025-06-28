import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load and preprocess data
@st.cache_data
def load_model():
    df = pd.read_csv("liver_cancer_prediction.csv")
    
    # Encode categorical columns
    cat_cols = ['Gender', 'Alcohol_Consumption', 'Smoking_Status', 'Hepatitis_B_Status',
                'Hepatitis_C_Status', 'Obesity', 'Diabetes', 'Rural_or_Urban',
                'Seafood_Consumption', 'Herbal_Medicine_Use', 'Healthcare_Access',
                'Screening_Availability', 'Treatment_Availability', 'Liver_Transplant_Access',
                'Preventive_Care']
    
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    X = df[['Population', 'Incidence_Rate', 'Mortality_Rate', 'Gender', 'Age', 'Alcohol_Consumption', 'Smoking_Status',
            'Hepatitis_B_Status', 'Hepatitis_C_Status', 'Obesity', 'Diabetes', 'Rural_or_Urban', 'Seafood_Consumption',
            'Herbal_Medicine_Use', 'Healthcare_Access', 'Screening_Availability', 'Treatment_Availability',
            'Liver_Transplant_Access', 'Preventive_Care', 'Survival_Rate', 'Cost_of_Treatment']]
    y = df['Prediction']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, le

model, le = load_model()

# Streamlit app UI
st.title("🩺 Liver Cancer Prediction App")
st.write("Enter patient details below to predict the likelihood of liver cancer.")

# Input fields
population = st.number_input("Population", min_value=0)
incidence = st.number_input("Incidence Rate", min_value=0.0)
mortality = st.number_input("Mortality Rate", min_value=0.0)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0)
alcohol = st.selectbox("Alcohol Consumption", ["Yes", "No"])
smoking = st.selectbox("Smoking Status", ["Yes", "No"])
hep_b = st.selectbox("Hepatitis B Status", ["Positive", "Negative"])
hep_c = st.selectbox("Hepatitis C Status", ["Positive", "Negative"])
obesity = st.selectbox("Obesity", ["Yes", "No"])
diabetes = st.selectbox("Diabetes", ["Yes", "No"])
location = st.selectbox("Rural or Urban", ["Rural", "Urban"])
seafood = st.selectbox("Seafood Consumption", ["Low", "Medium", "High"])
herbal = st.selectbox("Herbal Medicine Use", ["Yes", "No"])
access = st.selectbox("Healthcare Access", ["Poor", "Average", "Good"])
screening = st.selectbox("Screening Availability", ["Available", "Not Available"])
treatment = st.selectbox("Treatment Availability", ["Available", "Not Available"])
transplant = st.selectbox("Liver Transplant Access", ["Available", "Not Available"])
preventive = st.selectbox("Preventive Care", ["Available", "Not Available"])
survival = st.number_input("Survival Rate", min_value=0.0)
cost = st.number_input("Cost of Treatment", min_value=0.0)

# Mapping categorical text to encoded values (same order used by LabelEncoder)
input_data = pd.DataFrame([{
    "Population": population,
    "Incidence_Rate": incidence,
    "Mortality_Rate": mortality,
    "Gender": 1 if gender == "Male" else 0,
    "Age": age,
    "Alcohol_Consumption": 1 if alcohol == "Yes" else 0,
    "Smoking_Status": 1 if smoking == "Yes" else 0,
    "Hepatitis_B_Status": 1 if hep_b == "Positive" else 0,
    "Hepatitis_C_Status": 1 if hep_c == "Positive" else 0,
    "Obesity": 1 if obesity == "Yes" else 0,
    "Diabetes": 1 if diabetes == "Yes" else 0,
    "Rural_or_Urban": 0 if location == "Rural" else 1,
    "Seafood_Consumption": {"Low": 0, "Medium": 1, "High": 2}[seafood],
    "Herbal_Medicine_Use": 1 if herbal == "Yes" else 0,
    "Healthcare_Access": {"Poor": 0, "Average": 1, "Good": 2}[access],
    "Screening_Availability": 1 if screening == "Available" else 0,
    "Treatment_Availability": 1 if treatment == "Available" else 0,
    "Liver_Transplant_Access": 1 if transplant == "Available" else 0,
    "Preventive_Care": 1 if preventive == "Available" else 0,
    "Survival_Rate": survival,
    "Cost_of_Treatment": cost
}])

# Prediction
if st.button("Predict"):
    result = model.predict(input_data)
    st.success(f"Predicted Outcome: {'Liver Cancer Detected' if result[0]==1 else 'No Liver Cancer'}")
