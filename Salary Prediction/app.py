import gradio as gr
import joblib
import numpy as np
import plotly.graph_objects as go

# Load model and encoders
model = joblib.load("model.pkl")
le_role, le_edu, le_loc, le_size = joblib.load("encoders.pkl")

# Predict function with Plotly chart
def predict_salary(role, experience, education, location, company_size):
    role_encoded = le_role.transform([role])[0]
    education_encoded = le_edu.transform([education])[0]
    location_encoded = le_loc.transform([location])[0]
    company_size_encoded = le_size.transform([company_size])[0]

    input_data = np.array([[role_encoded, experience, education_encoded, location_encoded, company_size_encoded]])
    predicted_salary = model.predict(input_data)[0]

    # Average salary (you can compute this from your dataset if needed)
    avg_salary = 1200000

    # Create comparison chart
    fig = go.Figure(data=[
        go.Bar(name="Your Prediction", x=["You"], y=[predicted_salary], marker_color="green"),
        go.Bar(name="Industry Avg", x=["Average"], y=[avg_salary], marker_color="orange")
    ])
    fig.update_layout(title="💼 Salary Comparison", yaxis_title="INR", barmode="group")

    return f"💰 Predicted Salary: ₹{int(predicted_salary):,}", fig

# Class lists from encoders
roles = le_role.classes_.tolist()
educations = le_edu.classes_.tolist()
locations = le_loc.classes_.tolist()
company_sizes = le_size.classes_.tolist()

# Gradio UI
gr.Interface(
    fn=predict_salary,
    inputs=[
        gr.Dropdown(roles, label="Job Role"),
        gr.Slider(0, 30, value=2, step=1, label="Years of Experience"),
        gr.Dropdown(educations, label="Education"),
        gr.Dropdown(locations, label="Location"),
        gr.Dropdown(company_sizes, label="Company Size"),
    ],
    outputs=[
        gr.Textbox(label="Predicted Salary"),
        gr.Plot(label="Salary Comparison Chart")
    ],
    title="🔮 SalaryScope: Smart Salary Predictor",
    description="Get your salary prediction using AI and compare with industry average.",
    theme="soft"
).launch(share=True)

