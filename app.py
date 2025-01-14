# Install required libraries
!pip install streamlit pandas scikit-learn

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import streamlit as st

# Step 1: Data Collection
# For this example, we'll create a synthetic dataset
data = {
    'age': [25, 34, 45, 23, 35, 52, 46, 33, 26, 29],
    'gender': ['M', 'F', 'F', 'M', 'M', 'F', 'M', 'F', 'M', 'F'],
    'income': [50000, 60000, 80000, 45000, 70000, 90000, 85000, 62000, 48000, 52000],
    'browsing_time': [120, 80, 150, 90, 110, 200, 160, 100, 130, 140],
    'clicked_ad': [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
}
df = pd.DataFrame(data)

# Step 2: Data Preprocessing
# Encode categorical variables
df['gender'] = df['gender'].map({'M': 0, 'F': 1})

# Split the data into training and testing sets
X = df.drop(columns=['clicked_ad'])
y = df['clicked_ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'ROC AUC: {roc_auc:.2f}')

# Step 5: Streamlit App
st.title("Ad Targeting Predictor")

age = st.number_input("Enter Age:", min_value=18, max_value=100, step=1)
gender = st.selectbox("Select Gender:", options=["Male", "Female"])
income = st.number_input("Enter Income:", min_value=10000, max_value=1000000, step=1000)
browsing_time = st.number_input("Enter Browsing Time (in minutes):", min_value=0, max_value=1000, step=1)

if st.button("Predict"):
    gender_encoded = 0 if gender == "Male" else 1
    input_data = pd.DataFrame([[age, gender_encoded, income, browsing_time]],
                              columns=['age', 'gender', 'income', 'browsing_time'])

    prediction = model.predict(input_data)[0]
    result = "Clicked Ad" if prediction == 1 else "Did Not Click Ad"

    st.write(f"The user is predicted to: {result}")

