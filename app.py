
import streamlit as st
import pickle
import numpy as np

# Load the s model
with open('salary_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Frontend UI
st.title("💼 Employee Salary Prediction using ML")
st.image("https://img.freepik.com/free-vector/employees-analyzing-bar-graph-laptop-screen_74855-5285.jpg", width=600)

# Input fields
experience = st.slider("Years of Experience", 0, 20, 1)
test_score = st.slider("Test Score (out of 100)", 0, 100, 50)
interview_score = st.slider("Interview Score (out of 10)", 0, 10, 5)

# Predict Button
if st.button("Predict Salary"):
    features = np.array([[experience, test_score, interview_score]])
    prediction = model.predict(features)[0]
    st.success(f"Estimated Salary: ₹ {round(prediction):,}")