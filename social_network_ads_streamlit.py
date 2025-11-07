import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# Load dataset
df = pd.read_csv("Social_Network_Ads.csv")

# Drop User ID if present (no if-check needed)
df.drop(['User ID'], axis=1, errors='ignore', inplace=True)

# Convert Gender to numeric if the column exists
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Ensure target exists
if 'Purchased' not in df.columns:
    raise ValueError("Target column 'Purchased' not found in the CSV.")

# Prepare features and target
X = df.drop('Purchased', axis=1)
y = df['Purchased']

# Train/test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate and print accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {acc * 100:.2f}%")

# Save model
with open("classification.pkl", "wb") as f:
    pickle.dump(model, f)

# Load model for Streamlit app
with open("classification.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("Social Network Ads — Purchase Predictor")
st.write("Enter details and click Predict.")

# --- Input fields ---
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=120, value=30, step=1)
salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=100.0, format="%.2f")

# Convert Gender to numeric
gender_val = 0 if gender == "Male" else 1

# --- Predict button ---
if st.button("Predict"):
    # Create input array in the same order as training
    X_input = np.array([[gender_val, age, salary]])

    # Make prediction
    pred = model.predict(X_input)[0]

    # Display result
    if int(pred) == 1:
        st.success("✅Prediction: Will Purchase")
    else:
        st.error("❌Prediction: Will Not Purchase")
