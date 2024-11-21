import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

# Load the data
df = pd.read_csv("data.csv")

# Split the data into features (X) and target (y)
X = df[['time_ms']]  # Ensure X is a 2D array (dataframe or 2D numpy array)
y = df['ax']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial features
degree = 10  # Set the desired polynomial degree
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

# Create the Plotly figure
fig = go.Figure()

# Add scatter plot of original data
fig.add_trace(go.Scatter(
    x=X['time_ms'], y=y,
    mode='markers',
    name='Original Data',
    marker=dict(size=8, color='blue', opacity=0.7)
))

# Add polynomial regression curve
fig.add_trace(go.Scatter(
    x=X_range.flatten(), y=y_range_pred,
    mode='lines',
    name=f'Polynomial Fit (Degree {degree})',
    line=dict(color='red', width=2)
))

# Customize layout
fig.update_layout(
    title="Polynomial Regression Fit",
    xaxis_title="time_ms",
    yaxis_title="ax",
    legend=dict(x=0.8, y=1.2),
    template="plotly_white",
    font=dict(size=14)
)

# Show the figure
fig.show()
