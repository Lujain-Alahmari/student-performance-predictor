# Import required libraries
import pandas as pd
from sklearn.linear_model import LinearRegression

# Create sample dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Score": [50, 55, 60, 65, 70, 78, 85, 92]
}

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Define input (study hours) and output (score)
X = df[["Study_Hours"]]
y = df["Score"]

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Get user input
hours = float(input("Enter study hours: "))

# Predict score based on study hours
prediction = model.predict(
    pd.DataFrame({"Study_Hours": [hours]})
)

# Display prediction result
print(f"Predicted Score: {prediction[0]:.2f}")