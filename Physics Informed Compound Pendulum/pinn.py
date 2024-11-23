import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data.csv")

X = df[["time_ms"]]
y = df["ax"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))


X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.hidden(x)

    def physics_loss(self, x, y_true):
        # Predict angular displacement (theta) from the neural network
        theta_pred = self.forward(x)

        # Compute first and second derivatives of theta with respect to time (x)
        dtheta_dx = torch.autograd.grad(
            theta_pred, x, grad_outputs=torch.ones_like(theta_pred), create_graph=True
        )[0]
        d2theta_dx2 = torch.autograd.grad(
            dtheta_dx, x, grad_outputs=torch.ones_like(dtheta_dx), create_graph=True
        )[0]

        # Pendulum parameters
        I = 1.499535266666667  # Moment of inertia
        c = 5.1234  # Damping coefficient
        m = 16.1096 / 1000  # Mass of pendulum bob
        g = 9.81  # Acceleration due to gravity
        r = 7.6194196337749736 / 100  # Length to center of mass

        # Compute horizontal acceleration a_x
        ax_pred = r * d2theta_dx2 * torch.cos(theta_pred) - r * (dtheta_dx**2) * torch.sin(theta_pred)

        # Physics constraint (pendulum equation with damping)
        physics_term = I * d2theta_dx2 + c * dtheta_dx + m * g * r * torch.sin(theta_pred)
        physics_loss1 = torch.mean(physics_term**2)

        # Data loss: Mean squared error between predicted and true horizontal acceleration
        mse_loss = torch.mean((ax_pred - y_true) ** 2)

        return mse_loss + physics_loss1




model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


num_epochs = 5000
loss_history = []

X_train_tensor.requires_grad = True

for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = model.physics_loss(X_train_tensor, y_train_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")

# Enable gradients for X_train_tensor and X_test_tensor
X_train_tensor.requires_grad = True
X_test_tensor.requires_grad = True

model.eval()

# Predictions for theta (keep in computation graph)
theta_pred_train = model(X_train_tensor)  # Output is tied to X_train_tensor
theta_pred_test = model(X_test_tensor)   # Output is tied to X_test_tensor

# First derivative: dtheta/dt
dtheta_train = torch.autograd.grad(
    theta_pred_train, X_train_tensor, grad_outputs=torch.ones_like(theta_pred_train), create_graph=True
)[0]
dtheta_test = torch.autograd.grad(
    theta_pred_test, X_test_tensor, grad_outputs=torch.ones_like(theta_pred_test), create_graph=True
)[0]

# Second derivative: d2theta/dt2
d2theta_train = torch.autograd.grad(
    dtheta_train, X_train_tensor, grad_outputs=torch.ones_like(dtheta_train), create_graph=True
)[0]
d2theta_test = torch.autograd.grad(
    dtheta_test, X_test_tensor, grad_outputs=torch.ones_like(dtheta_test), create_graph=True
)[0]

# Pendulum parameters
r = 7.6194196337749736 / 100  # Length to center of mass

# Compute horizontal acceleration a_x
ax_pred_train = r * d2theta_train * torch.cos(theta_pred_train) - r * (dtheta_train**2) * torch.sin(theta_pred_train)
ax_pred_test = r * d2theta_test * torch.cos(theta_pred_test) - r * (dtheta_test**2) * torch.sin(theta_pred_test)

# Detach and convert to NumPy after all calculations
ax_pred_train = ax_pred_train.detach().numpy()
ax_pred_test = ax_pred_test.detach().numpy()

# Scale back to the original range
ax_pred_train = scaler_y.inverse_transform(ax_pred_train)
ax_pred_test = scaler_y.inverse_transform(ax_pred_test)

# Plotting
plt.figure(figsize=(12, 6))

# Scatter plots for training and testing data
plt.scatter(X_train, y_train, color="blue", label="Training Data", alpha=0.7)
plt.scatter(X_test, y_test, color="green", label="Testing Data", alpha=0.7)

# Sort indices for a smooth curve
sorted_indices_train = np.argsort(X_train.values.squeeze())
sorted_indices_test = np.argsort(X_test.values.squeeze())

# Predicted curves
plt.plot(
    X_train.values[sorted_indices_train],
    ax_pred_train[sorted_indices_train],
    color="red",
    linewidth=2,
    label="PINN Fit (Train)",
)
plt.plot(
    X_test.values[sorted_indices_test],
    ax_pred_test[sorted_indices_test],
    color="purple",
    linewidth=2,
    linestyle="--",
    label="PINN Fit (Test)",
)

# Plot settings
plt.xlabel("Time (ms)", fontsize=14)
plt.ylabel("Ax", fontsize=14)
plt.title("Physics-Informed Neural Network Fit vs Original Data", fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.5)
plt.show()
