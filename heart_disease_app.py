import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Heart Disease Prediction App", layout="centered")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

data = load_data()

# Sidebar Navigation
st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio("Go to", ["About App", "Prediction", "Dataset"])

# ---------- About Section ----------
if page == "About App":
    st.title("💓 Heart Disease Prediction App")
    st.markdown("""
    This app uses machine learning to predict whether a person is at risk of heart disease based on health inputs.  
    **Technologies:** Streamlit, Scikit-learn, Pandas, Random Forest  
    **Created by:** You 😎  
    """)
    st.info("Navigate using the sidebar to try a prediction or view the dataset.")

# ---------- Dataset Section ----------
elif page == "Dataset":
    st.title("📊 Heart Disease Dataset")
    with st.expander("Show full dataset"):
        st.dataframe(data)

    st.markdown("### Dataset Overview")
    st.write(data.describe())

# ---------- Prediction Section ----------
elif page == "Prediction":
    st.title("🧠 Heart Disease Predictor")

    # Layout input fields using columns
    with st.expander("Enter Patient Health Data"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 29, 77, 45)
            sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
            cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True)", [0, 1])
            restecg = st.selectbox("Resting ECG", [0, 1])

        with col2:
            trestbps = st.slider("Resting Blood Pressure", 94, 200, 120)
            chol = st.slider("Cholesterol", 126, 564, 200)
            thalach = st.slider("Max Heart Rate Achieved", 71, 202, 150)
            exang = st.selectbox("Exercise-Induced Angina", [0, 1])
            oldpeak = st.slider("ST depression (Oldpeak)", 0.0, 6.2, 1.0)

        with col3:
            slope = st.selectbox("Slope of ST segment", [0, 1, 2])
            ca = st.slider("Number of vessels colored by fluoroscopy", 0, 4, 0)
            thal = st.selectbox("Thalassemia (0–2)", [0, 1, 2])

    # Model input
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]],
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                       'restecg', 'thalach', 'exang', 'oldpeak',
                                       'slope', 'ca', 'thal'])

    # Train model
    X = data.drop("target", axis=1)
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predict
    if st.button("🔍 Predict"):
        prediction = model.predict(input_data)[0]
        pred_prob = model.predict_proba(input_data)[0][prediction]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🚨 **Risk Detected!** The patient might be at risk of heart disease.")
            st.markdown(f"**Prediction Confidence:** {pred_prob:.2%}")
            st.markdown("⚠️ Consider consulting a doctor for further tests.")
        else:
            st.success("✅ **No Risk Detected.** The patient is likely healthy.")
            st.markdown(f"**Prediction Confidence:** {pred_prob:.2%}")
            st.markdown("😃 Keep maintaining a healthy lifestyle!")

    st.sidebar.markdown("### Model Accuracy")
    st.sidebar.info(f"{model.score(X_test, y_test) * 100:.2f}%")
