import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load Dataset
data = pd.read_csv("student_scores.csv")

# Features and Target
X = data[['Hours_Studied', 'Attendance', 'Previous_Score']]
y = data['Final_Score']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict Test Data
y_pred = model.predict(X_test)

# Evaluate Model
print("R2 Score:", round(r2_score(y_test, y_pred), 4))

# User Prediction
hours = float(input("Hours Studied: "))
attendance = float(input("Attendance %: "))
previous = float(input("Previous Score: "))

new_student = pd.DataFrame({
    'Hours_Studied': [hours],
    'Attendance': [attendance],
    'Previous_Score': [previous]
})

prediction = model.predict(new_student)

print("Predicted Final Score:", round(prediction[0], 2))

# Graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Scores")
plt.show()