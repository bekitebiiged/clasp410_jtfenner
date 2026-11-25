import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator
import pylab as p
from scipy.interpolate import interp1d
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from cycler import cycler
from matplotlib.patches import FancyArrowPatch
from collections.abc import Iterable
#Adjust plotting stuff :)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.facecolor'] = 'gainsboro'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['legend.facecolor'] = 'oldlace'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.grid'] = True
colors = ['teal','orchid', 'darkgoldenrod', 'firebrick' ]
linestyles = ['-', ':', '--', '-.']
custom_cycler = (cycler(color=colors) +
                 cycler(linestyle=linestyles)
)
plt.rc('axes', prop_cycle=custom_cycler)
DEBUG = False

x_grid_size = 1e3 #m
y_grid_size = 1e3 #m

x_size = 20e3 #m
y_size = 10e3 #m

#assuming that we are on a flat plane


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
print(x.shape, y.shape)

west_values = [np.array([0, 3, 6, 8])*1e3, np.array([967, 985, 1000, 990])]
east_values = [np.array([1, 4, 9])*1e3, np.array([965, 995, 970])]

west_interp = interp1d(west_values[0], west_values[1], kind = 'linear', fill_value='extrapolate')(y)
east_interp = interp1d(east_values[0], east_values[1], kind = 'linear', fill_value='extrapolate')(y)

#print(f"Eastern Boundary Interpolation: {east_interp}")
#print(f"Western Boundary Interpolation: {west_interp}")

# Next include the north/south boundary interpolation
# using the interpolated values for east and western boundaries
north_values = [np.array([x[0], x[-1]]), np.array([west_interp[-1], east_interp[-1]])]
south_values = [np.array([x[0], x[-1]]), np.array([west_interp[0],east_interp[0]])]

north_interp = interp1d(north_values[0], north_values[1], kind='linear', fill_value='extrapolate')(x)
south_interp = interp1d(south_values[0], south_values[1], kind='linear', fill_value='extrapoalte')(x)

#print(f"Northern Boundary interpolation: {north_interp}\n")
#print(f"Southern Boundary interpolation: {south_interp}")

# Plot and check the boundary conditions
full_steady_state = np.zeros( (len(x), len(y)) )
full_steady_state[:, 0] = south_interp
full_steady_state[:, -1] = north_interp
full_steady_state[0, :] = west_interp
full_steady_state[-1, :] = east_interp

if DEBUG:
    plt.imshow(full_steady_state)
    plt.show()


Nx = len(x) - 2
Ny = len(y) - 2
x_Kform = [(np.ones(Nx)*-2), np.ones(Nx), np.ones(Nx)]
y_Kform = [(np.ones(Ny)*-2), np.ones(Ny), np.ones(Ny)]
Dxx = spdiags(x_Kform, [0, -1, 1], Nx, Nx) / x_grid_size**2
Dyy = spdiags(y_Kform, [0, -1, 1], Ny, Ny) / y_grid_size**2
#print(Dxx.toarray(), Dyy.toarray())
L = (sp.kron(sp.eye(Nx), Dyy) + sp.kron(Dxx, sp.eye(Ny)))
#plt.spy(L)
#plt.show()

# west, east, south, north interp
rhs_array = np.zeros( (Ny, Nx) )


# Using Fortran ordering! column-major order
rhs_array[:,0] = west_interp[1:-1] / x_grid_size ** 2
rhs_array[:,Nx-1] =  east_interp[1:-1] / x_grid_size ** 2
rhs_array[0,:] = rhs_array[0,:] + south_interp[1:-1] / y_grid_size ** 2 # Plus the South values
rhs_array[Ny-1, :] = rhs_array[Ny-1, :] + north_interp[1:-1] / y_grid_size ** 2 # Add the Northern values

# Stretch out our grid into a giant column vector
# Using Fortran order
rhs_column = rhs_array.ravel(order = 'F')

steady_state = -spsolve(L, rhs_column)
# Reshape the output to a matrix
steady_state = np.reshape(steady_state, (Ny, Nx), order = 'F')
if DEBUG:
    plt.imshow(steady_state)
    plt.show()
# Add the boundary values to the center
full_steady_state = np.pad(steady_state, pad_width=1, mode = 'constant', constant_values = 0)
full_steady_state[-1,  :] = north_interp
full_steady_state[0, :] = south_interp
full_steady_state[:, 0] = west_interp
full_steady_state[: , -1] = east_interp

if DEBUG:
    plt.imshow(full_steady_state)
    plt.show()
fig, ax = plt.subplots(1, 1, figsize = (6 ,4) )
X, Y = np.meshgrid(x, y)

color_contour = ax.contourf(X, Y, full_steady_state)
contour = ax.contour(X, Y, full_steady_state, colors = 'k', linestyles = '--')
fig.colorbar(color_contour, ax = ax, label = "Hydropotential ($m^2s^{-2}$)")

ax.set_ylabel("Northing (km)")
ax.set_xlabel("Easting (km)")
ax.set_title("Steady State Hydropotential")
plt.show()


# Now we will add the contanimation and model it's path!
fig, ax = plt.subplots(1, 1, figsize = (6 ,4) )
#u, v = -dZ*K*hydrogradient in x, -dZ*K*hydrogradiant in y
curr_state = full_steady_state
# find the gradient in x and y
hydro_y_grad, hydro_x_grad = np.gradient(curr_state, y_grid_size, x_grid_size)
contour = ax.contour(X, Y, curr_state, colors = 'k', linestyles = '--')
color_contour = ax.contourf(X, Y, full_steady_state)
fig.colorbar(color_contour, ax = ax, label = "Hydropotential ($m^2s^{-2}$)")
ax.quiver(X, Y, hydro_x_grad, hydro_y_grad)
plt.show()

print(hydro_y_grad.shape)
print(Y.shape)
print(Y[-1])
K = 0.2 #m/s
dZ = 1 #m
g = 9.8 #m/s
# interpolate the u,v to parcel grid
# convert x,y hydropoten_grad to head_grad
head_x_grad = hydro_x_grad / g # unitless (m^2s^-2 / m^2s^-2)
head_y_grad = hydro_y_grad / g # unitless (m^2s^-2 / m^2s^-2)
u = -(K * head_x_grad * dZ)
v = -(K * head_y_grad * dZ)

# put the u,v values onto the regular X,Y grid
u_interpolator = RegularGridInterpolator( (y, x), u)
v_interpolator = RegularGridInterpolator( (y, x), v)
X_speed = u_interpolator((Y,X), method = 'linear')
Y_speed = v_interpolator( (Y,X), method = 'linear')
fig, ax = plt.subplots(1, 1)
ax.quiver(X, Y, X_speed, Y_speed)
plt.show()


