import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# Load model
model = joblib.load("model.pkl")
le_role, le_edu, le_loc, le_size = joblib.load("encoders.pkl")

st.title("🔮 Employee Salary Predictor")

# Inputs
role = st.selectbox("Job Role", le_role.classes_)
experience = st.slider("Years of Experience", 0, 30, 2)
education = st.selectbox("Education", le_edu.classes_)
location = st.selectbox("Location", le_loc.classes_)
company_size = st.selectbox("Company Size", le_size.classes_)

if st.button("Predict Salary"):

    role_encoded = le_role.transform([role])[0]
    education_encoded = le_edu.transform([education])[0]
    location_encoded = le_loc.transform([location])[0]
    company_size_encoded = le_size.transform([company_size])[0]

    input_data = np.array([[role_encoded, experience, education_encoded, location_encoded, company_size_encoded]])
    predicted_salary = model.predict(input_data)[0]

    avg_salary = 1200000

    st.success(f"💰 Predicted Salary: ₹{int(predicted_salary):,}")

    fig = go.Figure(data=[
        go.Bar(name="Your Prediction", x=["You"], y=[predicted_salary]),
        go.Bar(name="Industry Avg", x=["Average"], y=[avg_salary])
    ])

    st.plotly_chart(fig)
