import numpy as np
import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Calculate and plot deflection of a cantilever beam with a force applied at any position.")

    parser.add_argument('--length', type=float, default=5.0, help="Length of the beam in meters (default: 5.0)")
    parser.add_argument('--youngs_modulus', type=float, default=210e9, help="Young's Modulus in Pascals (default: 210e9 for steel)")
    parser.add_argument('--second_moment', type=float, default=0.0001, help="Second moment of area in meters^4 (default: 0.0001)")
    parser.add_argument('--force', type=float, default=100000.0, help="Force applied on the beam in Newtons (default: 100000.0)")
    parser.add_argument('--position', type=float, default=3.0, help="Position along the beam where force is applied in meters (default: 3.0)")

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    L = args.length          
    E = args.youngs_modulus 
    I = args.second_moment   
    F = args.force           
    a = args.position   

    x_vals = np.linspace(0, L, 100)

    deflection = -np.piecewise(
        x_vals,
        [x_vals <= a, x_vals > a],
        [
            lambda x: (F * x**2) / (6 * E * I) * (3 * a - x),
            lambda x: (F * a**2) / (6 * E * I) * (3 * x - a)
        ]
    )
    
    np.savez('cantilever_beam_deflection.npz', x_vals=x_vals, deflection=deflection)
    
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, deflection, label="Deflection Curve", color="b")
    plt.xlabel("Position along the beam (m)")
    plt.ylabel("Deflection (m)")
    plt.title("Cantilever Beam Deflection with Applied Force at Any Position")

    plt.plot(x_vals, np.zeros_like(x_vals), 'k--', label="Original Beam Position")

    deflection_at_a = -(F * a**2) / (6 * E * I) * (3 * a - a)
    plt.arrow(a, deflection_at_a, 0, -0.2, head_width=0.1, head_length=0.05, fc='r', ec='r', label="Applied Force (F)")
    plt.text(a + 0.1, -0.22, f"F = {F}N", color="r", fontsize=12, ha='center')

    plt.legend()

    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()

