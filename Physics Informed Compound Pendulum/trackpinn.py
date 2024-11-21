import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

# Scale features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, requires_grad=True)
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
        physics_term = d2ydx2 + dydx + 2 * y_pred
        physics_loss = torch.mean(physics_term ** 2)
        mse_loss = torch.mean((y_pred - y_true) ** 2)
        return mse_loss + physics_loss


model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 5000
loss_history = []
predictions_history = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = model.physics_loss(X_train_tensor, y_train_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    with torch.no_grad():
        predictions_history.append(model(X_train_tensor).detach().numpy())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")

sorted_indices = np.argsort(X_train.values.squeeze())
X_sorted = X_train.values[sorted_indices].squeeze()
y_sorted = y_train.values[sorted_indices]

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(X_train, y_train, color="blue", label="Training Data", alpha=0.7)
(line,) = ax.plot([], [], color="red", label="PINN Fit", linewidth=2)
ax.set_xlabel("Time (ms)", fontsize=14)
ax.set_ylabel("Ax", fontsize=14)
ax.set_title("PINN Training Progress", fontsize=16)
ax.legend(fontsize=12)
ax.grid(alpha=0.5)


def update(frame):
    current_prediction = scaler_y.inverse_transform(
        predictions_history[frame][sorted_indices]
    )
    line.set_data(X_sorted, current_prediction)
    ax.set_title(f"PINN Training Progress: Epoch {frame + 1}/{num_epochs}")
    return (line,)


# Create animation
ani = FuncAnimation(fig, update, frames=num_epochs, interval=50, blit=True)

plt.show()
