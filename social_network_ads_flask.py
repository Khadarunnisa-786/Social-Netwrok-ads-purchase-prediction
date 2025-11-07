from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("classification.pkl", "rb") as f:
    model = pickle.load(f)

# Define the feature order (same as during training)
feature_order = ["Gender", "Age", "EstimatedSalary"]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Extract features in correct order
    gender = data.get("Gender", "").lower()
    age = float(data.get("Age", 0))
    salary = float(data.get("EstimatedSalary", 0))

    # Encode Gender (same way as training)
    if gender == "male":
        gender_val = 0
    elif gender == "female":
        gender_val = 1
    else:
        gender_val = 0  # default

    # Prepare input for model
    X = np.array([[gender_val, age, salary]])

    # Predict
    prediction = model.predict(X)[0]

    # Convert to readable text
    result = "✅Will Purchase" if prediction == 1 else "❌Will Not Purchase"

    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
