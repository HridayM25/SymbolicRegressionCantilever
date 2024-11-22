from gplearn.genetic import SymbolicRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('data.csv')
x_vals = data['time_ms'].to_numpy()
deflection = data['ax'].to_numpy()

# Reshape X for fitting
X = x_vals.reshape(-1, 1)
y = deflection  # y can remain 1D

# Normalize/Scale the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)  # Scale the input feature
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()  # Scale the target

# Initialize and fit the Symbolic Regressor
est = SymbolicRegressor(
    population_size=2000,  # Increased size for better exploration
    generations=50,        # Increased generations for more thorough search
    random_state=42
)
est.fit(X_scaled, y_scaled)

# Access and print the best program (expression)
best_expression = str(est)  # Preferred way
print("Best Expression:")
print(best_expression)

# Optionally, print the predicted values and the scaling factors for the original scale
y_pred_scaled = est.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()  # Inverse scale to get original deflection values

# Show the first few predicted values alongside actual values
print("Predicted vs Actual (scaled back to original deflection values):")
for true_val, pred_val in zip(y[:10], y_pred[:10]):
    print(f"True: {true_val}, Predicted: {pred_val}")
