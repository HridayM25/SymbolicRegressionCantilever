import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation


g = 9.81  
m = 1.0   
L = 2.0   
theta0 = np.pi / 4  
omega0 = 0.0       
dt = 0.01           
b = 0.05          

triangle_vertices = np.array([
    [-L / 2, -np.sqrt(3) * L / 6],
    [L / 2, -np.sqrt(3) * L / 6], 
    [0, np.sqrt(3) * L / 3]        
])

def calculate_centroid(vertices):
    return np.mean(vertices, axis=0)

def setup_pendulum(vertices, pivot_point):
    # Recompute vertices relative to the pivot
    relative_vertices = vertices - pivot_point

    CM = np.mean(relative_vertices, axis=0)
    
    I = m * (np.sum((relative_vertices)**2) / 3)
    
    return relative_vertices, CM, I

pivot_point = calculate_centroid(triangle_vertices) 

pivot_point = triangle_vertices[2]

# Recompute geometry based on the pivot
relative_vertices, CM, I = setup_pendulum(triangle_vertices, pivot_point)

t_max = 10.0
time = np.arange(0, t_max, dt)

theta = np.zeros_like(time)
omega = np.zeros_like(time)
theta[0] = theta0
omega[0] = omega0


for i in range(1, len(time)):

    r = np.linalg.norm(CM)  
    torque = -m * g * r * np.sin(theta[i - 1])
    
    damping_torque = -b * omega[i - 1]
    
    total_torque = torque + damping_torque

    alpha = total_torque / I

    omega[i] = omega[i - 1] + alpha * dt
    theta[i] = theta[i - 1] + omega[i] * dt

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-2 * L, 2 * L)
ax.set_ylim(-2 * L, 2 * L)
ax.set_title("Compound Pendulum: Solid Triangle with Variable Pivot")
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
