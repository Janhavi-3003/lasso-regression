import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

st.title("📊 Student Exam Score Predictor (Lasso Regression)")

# Load dataset
df = pd.read_csv("student_exam_scores.csv")

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

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
st.write(f"**MSE:** {mse}")
st.write(f"**R² Score:** {r2}")

# Feature importance
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lasso.coef_
})

st.subheader("🔍 Feature Importance")
st.dataframe(coefficients)
