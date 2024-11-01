from gplearn.genetic import SymbolicRegressor
import numpy as np
import graphviz

data = np.load('cantilever_beam_deflection.npz')
x_vals = data['x_vals']
deflection = data['deflection']

X = x_vals.reshape(-1,1)
y = deflection.reshape(-1,1)

est = SymbolicRegressor(population_size=1000, generations=10)

est.fit(X, y)

best_expression = est._program
print(best_expression)