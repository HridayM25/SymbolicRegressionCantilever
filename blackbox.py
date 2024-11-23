import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("data.csv")

# Split the data into features (X) and target (y)
X = df[['time_ms']]  # Ensure X is a 2D array (dataframe or 2D numpy array)
y = df['ax']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial features
degree = 3  # Set the desired polynomial degree
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train the polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Check the model's performance
y_pred = poly_model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Predict for a new data point
new_data = pd.DataFrame({'time_ms': [100]})
new_data_poly = poly.transform(new_data)
prediction = poly_model.predict(new_data_poly)
print(f"Prediction: {prediction[0]:.2f}")

# Prepare data for plotting
X_range = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)  # Generate a smooth range of X values
X_range_poly = poly.transform(X_range)
y_range_pred = poly_model.predict(X_range_poly)

# Create the Matplotlib figure
plt.figure(figsize=(8, 6))

# Scatter plot of original data
plt.scatter(X['time_ms'], y, color='blue', s=30, label='Original Data', alpha=0.7)

# Polynomial regression curve
plt.plot(X_range.flatten(), y_range_pred, color='red', linewidth=2, label=f'Polynomial Fit (Degree {degree})')

# Customize the plot
plt.title("Polynomial Regression Fit")
plt.xlabel("time_ms")
plt.ylabel("ax")
plt.legend(loc='upper left')
plt.grid(True)

# Show the plot
plt.show()

# 5 -? 0.24
# 4 -> 0.22
# 3-> 0.22