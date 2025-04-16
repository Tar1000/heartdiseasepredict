# heart_disease_app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("Tar Joel💓Heart Disease Prediction App")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

data = load_data()
st.subheader("Sample of Dataset")
st.write(data.head())

# Feature and target separation
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction form
st.sidebar.header("Patient Input Features")

def user_input_features():
    age = st.sidebar.slider('Age', 29, 77, 50)
    sex = st.sidebar.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
    cp = st.sidebar.slider('Chest Pain Type (0-3)', 0, 3, 1)
    trestbps = st.sidebar.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.sidebar.slider('Cholesterol (mg/dl)', 100, 600, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)', [0, 1])
    restecg = st.sidebar.slider('Resting ECG (0-2)', 0, 2, 1)
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 60, 210, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (1 = yes; 0 = no)', [0, 1])
    oldpeak = st.sidebar.slider('ST depression induced by exercise', 0.0, 6.2, 1.0)
    slope = st.sidebar.slider('Slope of peak exercise ST segment (0-2)', 0, 2, 1)
    ca = st.sidebar.slider('Major vessels colored by fluoroscopy (0-4)', 0, 4, 0)
    thal = st.sidebar.slider('Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)', 0, 3, 1)

    return pd.DataFrame({
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
        'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
        'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
    })

input_df = user_input_features()
st.subheader("Patient Input")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction Result")
st.write("🟢 **Heart Disease Detected**" if prediction[0] == 1 else "🟡 **No Heart Disease Detected**")
st.write("Prediction Probability:", prediction_proba)

# Model Evaluation
st.subheader("Model Evaluation on Test Set")
y_pred = model.predict(X_test)

st.write("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))
