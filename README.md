# Social Network Ads Purchase Prediction

## 📌 Description
This project predicts whether a user will purchase a product based on social media advertisement data.

## ❓ Problem Statement
Companies want to identify potential customers who are likely to purchase products after seeing ads.

## 💡 Solution
Built a Machine Learning model using Random Forest Classifier to predict user purchase behavior based on features like Age, Gender, and Estimated Salary.

## 🛠️ Tools & Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Flask

## 📊 Results
Achieved good accuracy in predicting whether a user will purchase a product.

## 🖼️ Output
(Add your Streamlit UI screenshot here)

## ▶️ How to Run
1. Install required libraries  
2. Run Flask or Streamlit file  
3. Enter user details to get prediction




   🖼️ Output
- The app predicts:
✅ “Will Purchase”
❌ “Will Not Purchase”
Displays prediction instantly via Streamlit and flask web apps



⚙️ Implementation
- Streamlit:
Imported all required libraries for data processing, modeling, and UI creation.
Loaded and cleaned the Social Network Ads dataset.
Applied preprocessing techniques to prepare the data.
Trained a Random Forest Classifier and checked its accuracy.
Saved the trained model using pickle for later use.
Built a Streamlit web app to take user input, display predictions, and view results in a web browser.

-Flask:
Imported required libraries like Flask, pickle, and numpy.
Loaded the pre-trained model (classification.pkl).
Created a Flask app with a /predict API endpoint.
Took JSON input (Gender, Age, Estimated Salary) and processed it.
Used the model to predict and returned the result as JSON, tested using Postman.


❓ Problem Statement
-  The goal is to predict whether a user will purchase a product after viewing an advertisement on a social networking platform.
The prediction is made based on the user’s Age, Gender, and Estimated Salary.


