import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('/content/dataset.csv')

# Features and target variable
X = data[['Freq [GHz]', 'g1(mm)', 'w1(mm)', 'Dielectric Constant']]
y = data['S(1,1) in dB']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Example prediction
example_input = np.array([[20.5, 1.8, 0.7, 2.2]])  # Replace with your input values
predicted_value = model.predict(example_input)
print(f"Predicted S(1,1) in dB: {predicted_value[0]}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Prediction')
plt.xlabel('Actual S(1,1) in dB')
plt.ylabel('Predicted S(1,1) in dB')
plt.title('Actual vs Predicted S(1,1) in dB')
plt.legend()
plt.grid()
plt.show()