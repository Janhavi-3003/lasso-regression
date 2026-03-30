import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("📊 Student Exam Score Predictor (Lasso Regression)")

# Upload or default dataset
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("student_exam_scores.csv")

# Show dataset
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# Check required columns
required_cols = ["hours_studied", "attendance_percent", "sleep_hours", "previous_scores", "exam_score"]

if not all(col in df.columns for col in required_cols):
    st.error("⚠️ Dataset must contain: " + ", ".join(required_cols))
else:
    # Features & target
    X = df[["hours_studied", "attendance_percent", "sleep_hours", "previous_scores"]]
    y = df["exam_score"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    lasso = Lasso(alpha=0.5)
    lasso.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = lasso.predict(X_test_scaled)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Model Evaluation")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # Feature importance
    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": lasso.coef_
    })

    st.subheader("🔍 Feature Importance")
    st.dataframe(coefficients)

    # -------- Prediction Section --------
    st.subheader("🎯 Predict Student Score")

    h = st.slider("Hours Studied", 0.0, 12.0, 5.0)
    a = st.slider("Attendance %", 0.0, 100.0, 75.0)
    s = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    p = st.slider("Previous Score", 0.0, 100.0, 60.0)

    if st.button("Predict Score"):
        input_data = [[h, a, s, p]]
        input_scaled = scaler.transform(input_data)
        prediction = lasso.predict(input_scaled)

        st.success(f"📊 Predicted Exam Score: {prediction[0]:.2f}")
