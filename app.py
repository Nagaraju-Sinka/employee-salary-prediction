import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

st.title("Employee Income Classifier")

age = st.number_input("Age", 17, 75, step=1)
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov', 'Not listed', 'State-gov', 'Self-emp-inc', 'Federal-gov'])
marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-AF-spouse'])
occupation = st.selectbox("Occupation", ['Exec-managerial', 'Prof-specialty', 'Craft-repair', 'Sales', 'Adm-clerical', 'Others'])
relationship = st.selectbox("Relationship", ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])
# race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
gender = st.selectbox("Gender", ['Male', 'Female'])

fnlwgt = 200000
education_num = 10
capital_gain = 0
capital_loss = 0
hours_per_week = 40
native_country = 'United-States'

encodings = {
    'relationship': {'Husband': 0, 'Not-in-family': 1, 'Own-child': 2, 'Unmarried': 3, 'Wife': 4, 'Other-relative': 5},
    'workclass': {'Private': 0, 'Self-emp-not-inc': 1, 'Local-gov': 2, 'Not listed': 3, 'State-gov': 4, 'Self-emp-inc': 5, 'Federal-gov': 6},
    'marital-status': {'Married-civ-spouse': 0, 'Never-married': 1, 'Divorced': 2, 'Separated': 3, 'Widowed': 4, 'Married-AF-spouse': 5},
    'occupation': {'Exec-managerial': 0, 'Prof-specialty': 1, 'Craft-repair': 2, 'Sales': 3, 'Adm-clerical': 4, 'Others': 5},
    'race': {'White': 0, 'Black': 1, 'Asian-Pac-Islander': 2, 'Amer-Indian-Eskimo': 3, 'Other': 4},
    'gender': {'Male': 0, 'Female': 1},
    'native-country': {'United-States': 0, 'Mexico': 1, 'Philippines': 2, 'Germany': 3, 'Canada': 4, 'Others': 5}
}

default_race = 'White'

input_data = np.array([[
    age,
    fnlwgt,
    education_num,
    encodings['workclass'][workclass],
    encodings['marital-status'][marital_status],
    encodings['occupation'][occupation],
    encodings['relationship'][relationship],
    encodings['race'][default_race],
    encodings['gender'][gender],
    capital_gain,
    capital_loss,
    hours_per_week,
    encodings['native-country'][native_country]
]])

if st.button("Predict Income"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Income: {prediction}")
