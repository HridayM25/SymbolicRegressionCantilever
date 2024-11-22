import numpy as np
import pandas as pd
from skopt import gp_minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def get_acceleration_time_series(damping_coeff, t_max=0.9, dt=0.01):
    g = 9.81  
    m = 16.1096 / 1000  
    theta0 = 0 
    omega0 = 0.0
    
    a, b_side, c = 13.0, 10.5, 6.0
    
    x2 = (b_side**2 - c**2 + a**2) / (2 * a)
    y2 = np.sqrt(b_side**2 - x2**2)
    triangle_vertices = np.array([
        [0, 0],          
        [a, 0],          
        [x2, y2]         
    ])
    
    relative_vertices = triangle_vertices - triangle_vertices[0]
    CM = np.mean(relative_vertices, axis=0)
    I = m * np.sum((relative_vertices)**2) / 3
    r = np.linalg.norm(CM)

    time = np.arange(0, t_max, dt)
    time = time
    theta = np.zeros_like(time)
    omega = np.zeros_like(time)
    alpha = np.zeros_like(time) 
    ax = np.zeros_like(time)     
    
    theta[0] = theta0
    omega[0] = omega0
    
    for i in range(1, len(time)):
        torque = -m * g * r * np.sin(theta[i - 1] + np.pi/2)
        damping_torque = -damping_coeff * omega[i - 1]
        
        alpha[i-1] = (torque + damping_torque) / I
        
        # a_x = -r * (alpha * sin(theta) + omega^2 * cos(theta))
        ax[i-1] = -r * (alpha[i-1] * np.sin(theta[i-1]) + 
                        omega[i-1]**2 * np.cos(theta[i-1]))
        
        omega[i] = omega[i - 1] + alpha[i-1] * dt
        theta[i] = theta[i - 1] + omega[i] * dt
    
    torque_final = -m * g * r * np.sin(theta[-1] + np.pi/2)
    damping_torque_final = -damping_coeff * omega[-1]
    alpha[-1] = (torque_final + damping_torque_final) / I
    ax[-1] = -r * (alpha[-1] * np.sin(theta[-1]) + 
                   omega[-1]**2 * np.cos(theta[-1]))
    
    return time, ax

# Example usage:
# if __name__ == "__main__":
#     # Test with different damping coefficients
#     dampings = [0.01, 0.05, 0.1]
    
#     plt.figure(figsize=(10, 6))
#     for damping in dampings:
#         t, acc = get_acceleration_time_series(damping)
#         plt.plot(t, acc, label=f'Damping = {damping}')
    
#     plt.xlabel('Time (s)')
#     plt.ylabel('X Acceleration (m/s²)')
#     plt.title('X Acceleration vs Time for Different Damping Coefficients')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    
    
def objective_function(damping_coeff, test_time, test_acceleration):
    """
    Calculate error between predicted and test acceleration
    """
    pred_time, pred_acceleration = get_acceleration_time_series(damping_coeff[0])

    test_time_seconds = test_time #/# 1000  
    
    interpolator = interp1d(pred_time, pred_acceleration, bounds_error=False, fill_value="extrapolate")
    pred_acceleration_interpolated = interpolator(test_time_seconds)
    
    mse = np.mean((pred_acceleration_interpolated - test_acceleration) ** 2)
    return mse

def find_optimal_damping_bayesian(test_time, test_acceleration):
    def objective(x):
        return objective_function([x[0]], test_time, test_acceleration)
    
    result = gp_minimize(
        objective,
        [(0.001, 10.0)],  
        n_calls=45,
        noise=0.45634,
        random_state=42
    )
    
    return result.x[0]

if __name__ == "__main__":

    
    data = pd.read_csv('data.csv')
    q1 = data['ax'].quantile(0.25)  
    q3 = data['ax'].quantile(0.75)  
    iqr = q3 - q1  

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    filtered_data = data[(data['ax'] >= lower_bound) & (data['ax'] <= upper_bound)]

    test_time = filtered_data['time_ms'].values[:100]
    test_acceleration = filtered_data['ax'].values[:100]
    
    optimal_damping = find_optimal_damping_bayesian(test_time, test_acceleration)
    print(f"Optimal damping coefficient: {optimal_damping}")
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(test_time/1000, test_acceleration, 'b-', label='Test Data', alpha=0.6)
    
    pred_time, pred_acceleration = get_acceleration_time_series(optimal_damping)
    plt.plot(pred_time, pred_acceleration, 'r--', label='Predicted (Optimal)', linewidth=2)
    
    plt.xlabel('Time (s)')
    plt.ylabel('X Acceleration (m/s²)')
    plt.title(f'Acceleration Comparison (Optimal Damping = {optimal_damping:.4f})')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    final_error = objective_function([optimal_damping], test_time, test_acceleration)
    print(f"Final Mean Squared Error: {final_error}")