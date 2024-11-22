import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

# Physical constants
g = 9.81  
m = 16.1096 / 1000  
theta0 = 0  
omega0 = 0.0  
dt = 0.01  
damping = 0.5

a, b_side, c = 13.0, 10.5, 6.0

x2 = (b_side**2 - c**2 + a**2) / (2 * a)
y2 = np.sqrt(b_side**2 - x2**2)
triangle_vertices = np.array([
    [0, 0],          
    [a, 0],          
    [x2, y2]         
])

pivot_point = triangle_vertices[0]

def setup_pendulum(vertices, pivot):
    relative_vertices = vertices - pivot
    CM = np.mean(relative_vertices, axis=0)
    I = m * np.sum((relative_vertices)**2) / 3
    return relative_vertices, CM, I

relative_vertices, CM, I = setup_pendulum(triangle_vertices, pivot_point)

t_max = 10.0
time = np.arange(0, t_max, dt)

theta = np.zeros_like(time)
omega = np.zeros_like(time)
theta[0] = theta0
omega[0] = omega0

for i in range(1, len(time)):
    r = np.linalg.norm(CM)
    torque = -m * g * r * np.sin(theta[i - 1] + np.pi/2) 
    damping_torque = -damping * omega[i - 1]
    alpha = (torque + damping_torque) / I

    omega[i] = omega[i - 1] + alpha * dt
    theta[i] = theta[i - 1] + omega[i] * dt

# Animation setup
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-1.5 * a, 1.5 * a)
ax.set_ylim(-1.5 * a, 1.5 * a)
ax.set_title("Compound Pendulum: Scalene Triangle")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")

triangle_patch = Polygon(triangle_vertices, closed=True, color='blue', alpha=0.7)
ax.add_patch(triangle_patch)
pivot_marker, = ax.plot([], [], 'ro')

def update(frame):
    angle = theta[frame]
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    rotated_vertices = (rotation_matrix @ relative_vertices.T).T + pivot_point
    
    triangle_patch.set_xy(rotated_vertices)
    pivot_marker.set_data(pivot_point[0], pivot_point[1])
    return triangle_patch, pivot_marker

ani = FuncAnimation(fig, update, frames=len(time), interval=dt * 1000, blit=True)

plt.show()