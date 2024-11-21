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

        y_pred = self.forward(x)

        dydx = torch.autograd.grad(
            y_pred, x, grad_outputs=torch.ones_like(y_pred), create_graph=True
        )[0]
        d2ydx2 = torch.autograd.grad(
            dydx, x, grad_outputs=torch.ones_like(dydx), create_graph=True
        )[0]

        physics_term = d2ydx2 + dydx + 2 * y_pred  # Example ODE: y'' + y' + 2y = 0
        physics_loss = torch.mean(physics_term ** 2)
        mse_loss = torch.mean((y_pred - y_true) ** 2)
        return mse_loss + physics_loss


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

model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_tensor).detach().numpy()
    y_pred_test = model(X_test_tensor).detach().numpy()


y_pred_train = scaler_y.inverse_transform(y_pred_train)
y_pred_test = scaler_y.inverse_transform(y_pred_test)


plt.figure(figsize=(12, 6))


plt.scatter(X_train, y_train, color="blue", label="Training Data", alpha=0.7)
plt.scatter(X_test, y_test, color="green", label="Testing Data", alpha=0.7)


sorted_indices_train = np.argsort(X_train.values.squeeze())
sorted_indices_test = np.argsort(X_test.values.squeeze())
plt.plot(
    X_train.values[sorted_indices_train],
    y_pred_train[sorted_indices_train],
    color="red",
    linewidth=2,
    label="PINN Fit (Train)",
)
plt.plot(
    X_test.values[sorted_indices_test],
    y_pred_test[sorted_indices_test],
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
