import matplotlib.pyplot as plt
import numpy as np

#Create a debug feature for the code to print out a bunch of info!
debug = False

#Set the plotting styles:
#Update these to adjust plotting aesthetics
plt.rcParams['figure.titlesize'] = 22
plt.rcParams['figure.constrained_layout.use'] = False
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.facecolor'] = 'gainsboro'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['legend.facecolor'] = 'oldlace'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.grid'] = True

color1 = 'teal'
color2 = 'sandybrown'
#Create a 'tolerance' for the upper and lower bounds for each axis
#Used mainly for adjusting values and having a good h or vline for reference
tol = 0.25

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
def experiment1():
    #see effect of emissivity on surface temperature

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

def plot_nlayer_surftemp(num_layers, solar_irradiance, e, albedo, temp_marker = None, title = None):
    #create N-layer model

    #create number of layers, starting with 0 -> 10 (count numbers/ints)
    layers = range(0, num_layers+1, 1)
    #create empty surf_temp list
    surf_temps = []
    for layer in layers:
        temp = model_atmosphere(layer, solar_irradiance, e, albedo)
        surf_temps.append(temp[0])


    fig, ax = plt.subplots(1, 1, figsize = (8, 5))
    ax.plot(layers, surf_temps, color = color1)

    #Ensure the temp marker is indicated and a proper value (to have a dotted line marker!)
    if temp_marker is not None:
        if (isinstance(temp_marker, int) or isinstance(temp_marker, float)):
            ax.hlines(  temp_marker,
                        0,
                        num_layers+1,
                        label = f'{temp_marker} K',
                        color = 'black',
                        linestyles='--',
                        linewidth = 2
            )
        else:
            raise ValueError("temp_marker must be a float or int!")

    if title is not None:
        if isinstance(title, str):
            ax.set_title(title)
        else:
            raise ValueError("title must be a string")
    else:
        ax.set_title("N-Layer Atmosphere Surface Temperature")
    ax.set_xbound(-tol, num_layers + tol)
    ax.set_xticks(range(0,num_layers+1))

    ax.set_ylabel("Surface Temperature [K]")
    ax.set_xlabel("Number of Layers")
    ax.legend()
    plt.show()

def plot_temp_profile(num_layers, solar_irradiance, e, albedo, temp_marker = None, title = None):

    #create the model and get temps
    temp_prof = model_atmosphere(num_layers, solar_irradiance, e, 0)
    fig, ax = plt.subplots(1, 1, figsize = (8, 5))
    ax.plot(temp_prof, np.arange(num_layers+1), color = color1)

    #If the temp marker is indicated and a proper value
    if temp_marker is not None:
        if (isinstance(temp_marker, int) or isinstance(temp_marker, float)):
            ax.vlines(  temp_marker,
                        0,
                        num_layers+1,
                        label = f'{temp_marker} K',
                        color = 'black',
                        linestyles='--',
                        linewidth = 2
            )
        else:
            raise ValueError("temp_marker must be a float or int!")

    ax.set_title(f"{num_layers}-Layer Temperature Profile")
    ax.set_ylabel("Atmospheric Layer")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylim(-tol, num_layers + tol)
    ax.legend()
    plt.show()


def validation_tests():
    print(create_coef_matrix(num_layers=3))


#See Surface Temp based on N-Layers
plot_nlayer_surftemp(10, S_0, e = 0.225, albedo = 0, temp_marker = 288)

#Show Earth's Temp profile
plot_temp_profile(2, S_0, e = 0.255, albedo = 0, temp_marker = 288)

#Check out Venus's surface temperature based on N-Layers
plot_nlayer_surftemp(30, 2600, e = 1, albedo = 0, temp_marker= 700, title = "N-Layer Venus Atmosphere")

