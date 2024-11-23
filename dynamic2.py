import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import argparse
from scipy.integrate import solve_ivp

parser = argparse.ArgumentParser(description="Cantilever beam with damping in different media.")
parser.add_argument("--length", type=float, default=5.0, help="Length of the beam in meters")
parser.add_argument("--youngs_modulus", type=float, default=210e9, help="Young's modulus of the material in Pascals")
parser.add_argument("--moment_of_inertia", type=float, default=0.0001, help="Second moment of area in m^4")
parser.add_argument("--force", type=float, default=100000.0, help="Force applied on the beam in Newtons")
parser.add_argument("--mass", type=float, default=50.0, help="Effective mass of the beam in kg")
parser.add_argument("--damping", type=float, default=50.0, help="Damping coefficient in Ns/m")
args = parser.parse_args()

L = args.length
E = args.youngs_modulus
I = args.moment_of_inertia
F = args.force
m = args.mass  
c = args.damping  

k = 3 * E * I / L**3

# Initial conditions: initial displacement and velocity
y0 = F / k  # Static deflection due to initial force
v0 = 0  # Initial velocity

def equation_of_motion(t, y):
    displacement, velocity = y
    dydt = [velocity, (-c * velocity - k * displacement) / m]
    return dydt

t_max = 10
t_eval = np.linspace(0, t_max, 500)

solution = solve_ivp(equation_of_motion, [0, t_max], [y0, v0], t_eval=t_eval)

time = solution.t
displacement = solution.y[0]

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(0, L)
ax.set_ylim(-1.5 * y0, 1.5 * y0)
ax.set_title("Vibrating Cantilever Beam with Damping")
ax.set_xlabel("Beam Length (m)")
ax.set_ylabel("Displacement (m)")

line, = ax.plot([], [], 'b-', lw=4)
marker, = ax.plot([], [], 'ro', markersize=8, label="Tip displacement")
ax.legend()

def init():
    line.set_data([0, L], [0, 0])
    marker.set_data([L], [0])  # Fixed: Using sequence for both x and y
    return line, marker

def update(frame):
    current_disp = displacement[frame]
    line.set_data([0, L], [0, current_disp])
    marker.set_data([L], [current_disp])  # Fixed: Using sequence for both x and y
    return line, marker

ani = FuncAnimation(fig, update, frames=len(time), init_func=init, blit=True, interval=20)

# Save as GIF
ani.save('cantilever_beam.gif', writer='pillow', fps=30)

plt.show()