import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import io

st.set_page_config(page_title="Joel Demo Heart Disease Predictor", layout="wide")

# Caching the model and data
@st.cache_data
def load_data():
    data = pd.read_csv("heart.csv")
    return data

@st.cache_resource
def train_model():
    df = load_data()
    X = df.drop('target', axis=1)
    y = df['target']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, X.columns.tolist()

# Create PDF report
def create_pdf(result_df, user_name="User"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Demo Heart Disease Prediction Report", ln=True, align="C")
    pdf.ln(10)

    # Name and Date
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Name: {user_name}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%B %d, %Y')}", ln=True)
    pdf.ln(10)

    for col in result_df.columns:
        pdf.cell(200, 10, txt=f"{col}: {result_df[col].values[0]}", ln=True)

    return pdf.output(dest='S').encode('latin1')


# Load model
model, feature_names = train_model()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About", "Prediction", "Dataset", "Visualizations"])

# About Page
if page == "About":
    st.title("Joel Demo 💓Heart Disease Prediction App")
    st.write("""
    This app uses machine learning to predict the likelihood of heart disease based on user input.  
    ✅ Built with Streamlit and RandomForestClassifier  
    🔐 **Disclaimer**: This tool is not a substitute for professional medical advice.  
    🚫 No personal data is stored or shared.  
    """)

# Prediction Page
elif page == "Prediction":
    st.title("🩺 Predict Heart Disease Risk")
    user_name = st.text_input("Enter your name", "Anonymous")

    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, key="age")
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key="sex")
            cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3], key="cp")
            trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, key="trestbps")
        with col2:
            chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, key="chol")
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], key="fbs")
            restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2], key="restecg")
            thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, key="thalach")
        with col3:
            exang = st.selectbox("Exercise Induced Angina", [0, 1], key="exang")
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, step=0.1, key="oldpeak")
            slope = st.selectbox("Slope of ST", [0, 1, 2], key="slope")
            ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3], key="ca")
            thal = st.selectbox("Thalassemia (0=normal, 1=fixed defect, 2=reversible)", [0, 1, 2], key="thal")

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                                    exang, oldpeak, slope, ca, thal]],
                                  columns=feature_names)

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][prediction] * 100

        emoji = "✅" if prediction == 0 else "⚠️"
        message = "No Heart Disease Detected." if prediction == 0 else "Risk of Heart Disease Detected!"

        st.markdown(f"### {emoji} {message}")
        st.progress(prob / 100)
        st.write(f"**Confidence Level**: {prob:.2f}%")

        # Result DataFrame for export
        result_df = input_data.copy()
        result_df["Prediction"] = message
        result_df["Confidence"] = f"{prob:.2f}%"

        # Download PDF
        pdf_bytes = create_pdf(result_df, user_name)
        st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="heart_report.pdf", mime='application/pdf')

# Dataset Page
elif page == "Dataset":
    st.title("📁 Heart Dataset")
    data = load_data()
    st.dataframe(data)
    st.write("### Target Variable Distribution")
    st.bar_chart(data['target'].value_counts())

# Visualization Page
elif page == "Visualizations":
    st.title("📊 Data Visualizations")
    df = load_data()

    st.subheader("Feature Distributions")
    selected = st.multiselect("Choose features to visualize", df.columns[:-1], default=['age', 'sex', 'cp'])

    for feature in selected:
        st.write(f"**Distribution of {feature}**")
        fig, ax = plt.subplots()
        if df[feature].nunique() <= 10:
            df[feature].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            ax.set_ylabel('')
        else:
            sns.histplot(df[feature], kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)
