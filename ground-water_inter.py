import numpy as np
from scipy.interpolate import interp1d
x_grid_size = 1e3 #m
y_grid_size = 1e3 #m

x_size = 20e3 #m
y_size = 10e3 #m

well_info = [
    [ 0, 0, 967], #x, y, hydropotential m^2/s^2
    [ 0, 3, 985],
    [ 0, 6, 1000],
    [ 0, 8, 990],
    [ 20, 1, 965],
    [ 20, 4, 995],
    [ 20, 9, 970],
]

x = np.arange(0, x_size+x_grid_size, x_grid_size)
y = np.arange(0, y_size+y_grid_size, y_grid_size)

west_values = [np.array([0, 3, 6, 8])*1e3, np.array([967, 985, 1000, 990])*1e3]
east_values = [np.array([1, 4, 9])*1e3, np.array([965, 995, 970])*1e3]

west_interp = interp1d(west_values[0], west_values[1], kind = 'linear', fill_value='extrapolate')(y)
east_interp = interp1d(east_values[0], east_values[1], kind = 'linear', fill_value='extrapolate')(y)

print(f"Eastern Boundary Interpolation: {east_interp}")
print(f"Western Boundary Interpolation: {west_interp}")

# Next include the top/bottom functions