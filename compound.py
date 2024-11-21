import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

# Physical constants
g = 9.81  # Gravitational acceleration (m/s^2)
m = 1.0   # Mass of the triangle (kg)
L = 2.0   # Side length of the equilateral triangle (m)
theta0 = np.pi / 4  # Initial angle (45 degrees)
omega0 = 0.0        # Initial angular velocity (rad/s)
dt = 0.01           # Time step (s)
b = 0.05            # Damping coefficient (kg·m²/s)

# Define triangle geometry (equilateral triangle vertices)
triangle_vertices = np.array([
    [-L / 2, -np.sqrt(3) * L / 6],  # Bottom-left vertex
    [L / 2, -np.sqrt(3) * L / 6],   # Bottom-right vertex
    [0, np.sqrt(3) * L / 3]         # Top vertex
])

# Function to calculate centroid of the triangle
def calculate_centroid(vertices):
    return np.mean(vertices, axis=0)

# Function to calculate the center of mass and moment of inertia
def setup_pendulum(vertices, pivot_point):
    # Recompute vertices relative to the pivot
    relative_vertices = vertices - pivot_point
    
    # Compute the center of mass (average of vertices relative to pivot)
    CM = np.mean(relative_vertices, axis=0)
    
    # Moment of inertia about the pivot
    I = m * (np.sum((relative_vertices)**2) / 3)
    
    return relative_vertices, CM, I

# Choose the pivot point (default is centroid)
pivot_point = calculate_centroid(triangle_vertices)  # Default pivot is the centroid
# Uncomment the next line to try a different pivot point (e.g., top vertex)
pivot_point = triangle_vertices[2]

# Recompute geometry based on the pivot
relative_vertices, CM, I = setup_pendulum(triangle_vertices, pivot_point)

# Simulation parameters
t_max = 10.0
time = np.arange(0, t_max, dt)

# Initialize arrays for motion
theta = np.zeros_like(time)
omega = np.zeros_like(time)
theta[0] = theta0
omega[0] = omega0

# Simulate motion with Euler's method
for i in range(1, len(time)):
    # Torque due to gravity: τ = -m * g * r * sin(θ)
    r = np.linalg.norm(CM)  # Distance of CM from pivot
    torque = -m * g * r * np.sin(theta[i - 1])
    
    # Damping torque: τ_damping = -b * ω
    damping_torque = -b * omega[i - 1]
    
    # Total torque
    total_torque = torque + damping_torque
    
    # Angular acceleration
    alpha = total_torque / I
    
    # Update angular velocity and angle
    omega[i] = omega[i - 1] + alpha * dt
    theta[i] = theta[i - 1] + omega[i] * dt

# Animation setup
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-2 * L, 2 * L)
ax.set_ylim(-2 * L, 2 * L)
ax.set_title("Compound Pendulum: Solid Triangle with Variable Pivot")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")

# Create the triangle patch
triangle_patch = Polygon(triangle_vertices, closed=True, color='blue', alpha=0.7)

# Add triangle and pivot marker to the plot
ax.add_patch(triangle_patch)
pivot_marker, = ax.plot([], [], 'ro')  # Pivot point marker

def update(frame):
    # Rotate the triangle based on the current angle
    angle = theta[frame]
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    rotated_vertices = (rotation_matrix @ relative_vertices.T).T + pivot_point

    # Update the triangle's position and pivot marker
    triangle_patch.set_xy(rotated_vertices)
    pivot_marker.set_data(pivot_point[0], pivot_point[1])
    return triangle_patch, pivot_marker

# Animate
ani = FuncAnimation(fig, update, frames=len(time), interval=dt * 1000, blit=True)

plt.show()
