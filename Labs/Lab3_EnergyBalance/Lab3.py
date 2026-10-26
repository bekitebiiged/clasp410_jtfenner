import matplotlib.pyplot as plt
import numpy as np
import pprint as pprint

#Set the plotting styles:
#Update these to adjust plotting aesthetics
plt.style.use("Solarize_Light2")
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.facecolor'] = 'gainsboro'
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['legend.facecolor'] = 'oldlace'

#Create a debug feature for the code to print out a bunch of info!
debug = False

def create_coef_matrix(num_layers = 1, e = 1):
    '''
    Function which creates the matrix of coefficients for an atmopspheric energy balance diagram.
    ==============
        Inputs
    ==============
        num_layers:
            Number of atmospheric layers in the model
        e:
            Emissivity value of the atmosphere (each layer)
    ==============
        Returns
    ==============
        coefs:
            Coefficient Matrix for the number of layers
            Ordered from the surface to the final layer (row 0 is surface balance)

    Example:
        A = create_coef_matrix(num_layers = 3, e = 1)
        This will create the following matrix:
            [
            [-1.  1.  0.  0.]   #This is the surface layer
            [ 1. -2.  1.  0.]   #This is the first layer
            [ 0.  1. -2.  1.]   #This is the second layer
            [ 0.  0.  1. -2.]   #This is the third layer
            ]

    '''

    # Ensure that the number of layers is correct!
    if( (num_layers < 0) or  (not isinstance(num_layers, int)) ):
        raise ValueError("Number of Layers must be a positive integer!")

    #Create a diagonal matrix with -2e along the diagonal
    # help with getting the diagonal matrix!
    # https://stackoverflow.com/questions/58139494/how-can-i-create-a-diagonal-matrix-with-numpy
    diag = np.ones(num_layers+1) * -2 * e
    diag[0] = -1

    #create a matrix with a diag
    coefs = np.diag(diag, 0)
    #add the 1-above/below off-diagonal values
    coefs += np.diag(np.ones(num_layers) * e, 1)
    coefs += np.diag(np.ones(num_layers) * e, -1)

    return coefs


#create_coef_matrix(num_layers=3)


def model_atmosphere(num_layers, solar_irradiance, e, albedo):
    '''
    A*fluxes = b, solving for fluxes!
    '''

    stef_boltz = 5.6703e-8 #Stefan-Boltzmann
    # create the coefficient matrix 'A'
    A = create_coef_matrix(num_layers=num_layers, e=e)

    # since A moves from surface -> Nth Layer, we will
    # create b with the first value as solar irradiance
    # (and everything else zero)
    b = np.zeros(num_layers+1)
    b[0] = -(solar_irradiance * (1 - albedo) / 4) # -S/4 * (1 - alpha)
    if debug:
        print("A is: ", A)
        print("b is: ", b)

    #solve for the fluxes
    try:
        fluxes = np.linalg.solve(A, b)
    except Exception as ex:
        #have a no atmosphere energy balance if emissivity is 0
        print(ex)
        fluxes =[b[0] / A[0,0]]

    #Create empty temperature array
    temps = np.zeros(len(fluxes))

    #convert fluxes to temperatures
    for idx, flux in enumerate(fluxes):
        temp = (flux / stef_boltz) ** (1/4)
        temps[idx] = temp

    if debug:
        print("fluxes are: ", fluxes)
        print("temps are: ", temps)

    return temps


model_atmosphere(0, 1361, 1, 0)

S_0 = 1350 # Wm^-2

#We will now conduct experiment 1
# see effect of emissivity on surface temperature

#create list of emissivities 0 -> 1, 0.1 step
emissivities = np.arange(0,1.1,0.1)
#create empty list of surface temps
surf_temps = []

for e in emissivities:
    temp = model_atmosphere(1, S_0, e, 0)
    surf_temps.append(temp[0])

fig, ax = plt.subplots(1, 1, figsize = (8, 5))

ax.plot(surf_temps, emissivities)
ax.set_title("1-Layer Atmosphere Surface Temperature")
ax.set_ylabel("Emissivity")
ax.set_xlabel("Surface Temperature [K]")
plt.show()

#create N-layer model