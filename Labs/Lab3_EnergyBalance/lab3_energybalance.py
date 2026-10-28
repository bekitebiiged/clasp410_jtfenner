'''
This is the script for Lab3_Energy_Balance for CLIMATE410 in Fall 2026.

The code is split into a few sections:

    INITIAL THINGS:
Basic intial setup-related lines
    * Importing modules
    * DEBUG toggling
    * Setting plotting parameters
    * declaring some constants

    FUNCTIONS:
Here we create our modelling and plotting functions
These include:
    * create_coef_matrix()
    * model_atmosphere()
    * experiment1()             #used to answer part of question3
    * plot_nlevel_surftemp()
    * plot_temp_profile()

    VALIDATION:
Here we validate some of the functions and such
Uncomment the code to test

    TESTING:
This is where the plots are created and questions answered

Enjoy!~~~
'''
#    ╔═════╗             ╔═╗     ╔═╗             ╔═╗
#    ╚═╗ ╔═╝ ╔═══════╗ ╔═╝ ╚═╗   ╚═╝ ╔═══════╗   ║ ║
#      ║ ║   ║ ╔═══╗ ║ ╚═╗ ╔═╝   ╔═╗ ║ ╔═══╗ ║   ║ ║
#      ║ ║   ║ ║   ║ ║   ║ ║     ║ ║ ║ ║   ║ ║   ║ ║
#    ╔═╝ ╚═╗ ║ ║   ║ ║   ║ ╚═══╗ ║ ║ ║ ╚═══╝ ╚═╗ ║ ║
#    ╚═════╝ ╚═╝   ╚═╝   ╚═════╝ ╚═╝ ╚═════════╝ ╚═╝
#
#
#    ╔═════════╗ ╔═╗   ╔═╗ ╔═╗
#    ╚═══╗ ╔═══╝ ║ ║   ║ ║ ╚═╝ ╔═══════╗ ╔═══════╗   ╔═════╗
#        ║ ║     ║ ╚═══╝ ║ ╔═╗ ║ ╔═══╗ ║ ║ ╔═══╗ ║ ╔═╝   ╔═╝
#        ║ ║     ║ ╔═══╗ ║ ║ ║ ║ ║   ║ ║ ║ ║   ║ ║ ╚═══╗ ╚═╗
#        ║ ║     ║ ║   ║ ║ ║ ║ ║ ║   ║ ║ ║ ╚═══╝ ║ ╔═══╝ ╔═╝
#        ╚═╝     ╚═╝   ╚═╝ ╚═╝ ╚═╝   ╚═╝ ╚═════╗ ║ ╚═════╝
#                                          ╔═══╝ ║
#                                          ╚═════╝
import matplotlib.pyplot as plt
import numpy as np
#Create a DEBUG feature for the code to print out a bunch of info!
DEBUG = False
COEF_DEBUG = False

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
COLOR1 = 'teal'
COLOR2 = 'sandybrown'
#Create a 'TOLerance' for the upper and lower bounds for each axis
#Used mainly for adjusting values and having a good h or vline for reference
TOL = 0.25
#Declare some constants to be used later
S_0 = 1350 # Wm^-2
A = 0.3 #Estimate for Earth Albedo from NASA Fact Sheet (bond albedo)

#╔═══════╗ ╔═╗   ╔═╗ ╔═╗     ╔═╗ ╔═══════╗ ╔═════════╗ ╔═════╗ ╔═══════╗ ╔═╗     ╔═╗ ╔═══════╗
#║ ╔═════╝ ║ ║   ║ ║ ║ ╚═══╗ ║ ║ ║ ╔═════╝ ╚═══╗ ╔═══╝ ╚═╗ ╔═╝ ║ ╔═══╗ ║ ║ ╚═══╗ ║ ║ ║ ╔═════╝
#║ ╚═══╗   ║ ║   ║ ║ ║ ╔═╗ ╚═╝ ║ ║ ║           ║ ║       ║ ║   ║ ║   ║ ║ ║ ╔═╗ ╚═╝ ║ ║ ╚═════╗
#║ ╔═══╝   ║ ║   ║ ║ ║ ║ ╚═╗   ║ ║ ║           ║ ║       ║ ║   ║ ║   ║ ║ ║ ║ ╚═╗   ║ ╚═════╗ ║
#║ ║       ║ ╚═══╝ ║ ║ ║   ╚═╗ ║ ║ ╚═════╗     ║ ║     ╔═╝ ╚═╗ ║ ╚═══╝ ║ ║ ║   ╚═╗ ║ ╔═════╝ ║
#╚═╝       ╚═══════╝ ╚═╝     ╚═╝ ╚═══════╝     ╚═╝     ╚═════╝ ╚═══════╝ ╚═╝     ╚═╝ ╚═══════╝
def create_coef_matrix(num_layers = 1, e = 1):
    '''
    Function which creates the matrix of coefficients for an N-layer atmospheric energy balance.
    Matrix is created using the surface balance row and creating diagonals to piece together a
    symmetric matrix along the center diagonal.
    ==============
        Inputs
    ==============
        num_layers: int
            Number of atmospheric layers in the model
        e:  float [0, 1]
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
        Becomes more complicated if e < 1, which is shown in general in the accompanying lab report.
    '''

    # Ensure that the number of layers is correct!
    if( (num_layers < 0) or  (not isinstance(num_layers, int)) ):
        raise ValueError("Number of Layers must be a positive integer!")

    #Create a diagonal matrix with -2e along the diagonal
    # help with getting the diagonal matrix!
    # https://stackoverflow.com/questions/58139494/how-can-i-create-a-diagonal-matrix-with-numpy

    #create a list for the first row (surface-layer energy balance)
    first_row = [-1] #first value is -1 (always), as the Earth releases flux
    #loop over the number of columns/fluxes
    for col in range(1,num_layers+1):
        #calculate the flux based on the column
        #column refers to atmospheric layer 'index'/'flux'.
        #np.max is called because the value at 1 throws an error
        first_row.append( float(e * (1-e)**(np.max( (0,(col-1)) )) ) )
        #print info if debug is on
        if DEBUG:
            print(f"curr element: e * (1-e)^{(np.max( (0, (col-1))) )}")
    #print the first row if debug is on
    if DEBUG:
        print(f"FIRST ROW OF COEF IS : \n{first_row}")

    #create a list to hold each diagonal
    diag_list = []
    #interate over the number of diagonals to create
    for curr_diag_idx in range(num_layers+1):
        #first diag = 0
        #the current diagonal is equal to the first row's idx
        #curr_diag = [vector of one's, length dependant on idx]
        #each value is made to be the value of the surface balance at current idx
        curr_diag = np.ones(num_layers + 1 - curr_diag_idx) * first_row[curr_diag_idx]
        #the center diag is treated specially
        if curr_diag_idx == 0:
            #the values from 1 -> end are equal to -2e (first value is -1)
            curr_diag[1:] = curr_diag[1:] * 2 * e
        #all off-center diagonals
        else:
            #
            #multiply by another e (due to absorption from that layer)
            curr_diag[1:] = curr_diag[1:] * e
        #add the created diag to a list of diags
        diag_list.append(curr_diag)
        #print useful stuff to debug
        if DEBUG:
            print(f"current diagonal is: {curr_diag_idx}")
            print(curr_diag)
    #the coefficient matrix will have the center diag by the first diag in the list
    coefs = np.diag(diag_list[0], 0)
    #put the diagonals to together into the final coefficent matrix!
    for idx, diag in enumerate(diag_list):
        #skip the first diag (already added)
        if idx == 0:
            continue
        #place the current diagonal in the upper triangle
        coefs += np.diag(diag, idx)
        #place the current diagonal in the lower triangle
        coefs += np.diag(diag, -idx)
    #print the matrix if debug!
    if COEF_DEBUG:
        print(coefs)
    return coefs

def model_atmosphere(num_layers, solar_irradiance, e, albedo, sw_absorbing_layer = 0):
    '''
    This function takes parameters to model an energy balance,
    returning the temperatures at each layer.
    Fluxes are calculated and then converted into temperatures using Stefan-Boltzmann Law.
    ==============
        Inputs
    ==============
        num_layers: int
            Number of atmospheric layers in the model
        solar_irradiance: int
            Solar flux (input energy)
        e:  float [0, 1]
            Emissivity value of the atmosphere (each layer)
        albedo:  float [0, 1]
            Albedo of the absorbing layer (1-e_{for the shortwave band})
        sw_absorbing_layer = 0: int
            Layer which will absorb incoming shortwave (solar irradiance).
            Can be shifted from surface to arbitrary layer.

    ==============
        Returns
    ==============
        temps: list
            list of temperatures at each atmospheric layer (including surface).
            Moves from surface to top of atmosphere.

    Example:
        temps = model_atmosphere(4, S_0, e = 0.255, a = a)
        print(temps)
            [285.04693212 254.7878797  245.03990989 233.96641402 221.0510526 ]
            [surface -> 1st layer -> 2nd layer -> 3rd layer -> 4th layer (nth)]
        temps_nuclear_winter = model_atmosphere(5, S_0, e = 0.5, a = 0, sw_absorbing_layer = 5)
            this will model a 5-layer atmosphere with emissivity of 0.5 for all atmospheric layers.
            albedo is 0, indicating the absorbing layer will intake all sw radiation
            sw_absorbing_layer = 5 means the 5th layer will absorb all sw (not the surface!)
    '''
    stef_boltz = 5.6703e-8 #Stefan-Boltzmann
    # create the coefficient matrix 'A'
    coefs = create_coef_matrix(num_layers=num_layers, e=e)

    # since A moves from surface -> Nth Layer, we will
    # create b with the first value as solar irradiance
    # (and everything else zero)
    b = np.zeros(num_layers+1)
    b[sw_absorbing_layer] = -(solar_irradiance * (1 - albedo) / 4) # -S/4 * (1 - alpha)
    if DEBUG:
        print("A is: ", coefs)
        print("b is: ", b)

    #solve for the fluxes
    try:
        fluxes = np.linalg.solve(coefs, b)
    except np.linalg.LinAlgError as ex:
        #have a no atmosphere energy balance if emissivity is 0
        if DEBUG:
            ex.add_note("Emissivity is 0 (no atmosphere)")
        fluxes =[b[0] / coefs[0,0]]

    #Create empty temperature array
    temps = np.zeros(len(fluxes))

    #convert fluxes to temperatures
    for idx, flux in enumerate(fluxes):
        temp = (flux / stef_boltz) ** (1/4)
        temps[idx] = temp

    if DEBUG:
        print("fluxes are: ", fluxes)
        print("temps are: ", temps)

    return temps

#We will now conduct experiment 1
def experiment1():
    '''
    This function looks at the emissivity,
        and how it affects the surface temperature of earth
        (1-layer model)
    It returns nothing but plots a emissivity vs surf_temp chart.
    '''
    #see effect of emissivity on surface temperature

    #create list of emissivities 0 -> 1, 0.1 step
    emissivities = np.arange(0,1.1,0.1)
    #create empty list of surface temps
    surf_temps = []

    for e in emissivities:
        temp = model_atmosphere(1, S_0, e, 0.3)
        surf_temps.append(temp[0])

    _, ax = plt.subplots(1, 1, figsize = (8, 5))
    ax.plot(emissivities, surf_temps)
    ax.hlines(  288,
                -TOL,
                1+TOL,
                label = '288 K',
                color = 'black',
                linestyles='--',
                linewidth = 2
            )
    ax.set_xlim(-0.01, 1+0.01)
    ax.set_title("1-Layer Atmosphere Surface Temperature")
    ax.set_xlabel("Emissivity")
    ax.set_ylabel("Surface Temperature [K]")
    ax.legend(loc = 'best')
    plt.show()

def plot_nlayer_surftemp(num_layers, solar_irradiance, e, albedo, temp_marker = None, title = None):
    '''
    This function plots calculates surface temperature
    based on given parameters up to a certain number of layers.
    number of layers on x-axis, surface temperature on y-axis.
    Optional ability to add the a dashed line to indicate a specified temperature (temp_marker)
    User can specify 'title' as well, but defaults to "N-Layer Atmosphere Surface Temperature"
    '''
    #create N-layer model

    #create number of layers, starting with 0 -> 10 (count numbers/ints)
    layers = range(0, num_layers+1, 1)
    #create empty surf_temp list
    surf_temps = []
    for layer in layers:
        temp = model_atmosphere(layer, solar_irradiance, e, albedo)
        surf_temps.append(temp[0])

    _, ax = plt.subplots(1, 1, figsize = (8, 5))
    ax.plot(layers, surf_temps, color = COLOR1)

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
    ax.set_xbound(-TOL, num_layers + TOL)

    if num_layers > 100:
        ax.set_xticks(range(0,num_layers+1, 10))
    elif num_layers > 10:
        ax.set_xticks(range(0,num_layers+1, 5))
    else:
        ax.set_xticks(range(0,num_layers+1))

    ax.set_ylabel("Surface Temperature [K]")
    ax.set_xlabel("Number of Layers")

    if temp_marker is not None:
        ax.legend(loc = 'best')

    plt.show()

def plot_temp_profile(num_layers, solar_irradiance, e, albedo,
                      temp_marker = None, title = None, sw_absorbing_layer = 0):
    '''
    This function plots calculates temperature profile
        based on given parameters and plots the temperature profile.

    Temperature on x-axis, atmospheric layer on y-axis.
        (on y-axis because height is up-down, feels right)
    Optional ability to add a dashed line to indicate a specified temperature (temp_marker)
    User can specify 'title' as well, but defaults to "N-Layer Atmosphere Surface Temperature"
        (N is the specified number of layers)
    User can also specify sw_absorbing_layer.
        (used for the nuclear winter scenario)
    '''

    #create the model and get temps
    temp_prof = model_atmosphere(num_layers, solar_irradiance, e, albedo,
                                sw_absorbing_layer=sw_absorbing_layer)
    #plot the modeled atmosphere
    _, ax = plt.subplots(1, 1, figsize = (8, 5))
    ax.scatter(temp_prof, np.arange(num_layers+1), color = COLOR1, zorder = 5)
    ax.plot(temp_prof, np.arange(num_layers+1), color = COLOR1,
            linestyle = '--', linewidth = 2, zorder = 5)

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
            ax.legend(loc = 'best')
        else:
            raise ValueError("temp_marker must be a float or int!")

    if title is not None and isinstance(title, str):
        ax.set_title(title)
    else:
        ax.set_title(f"{num_layers}-Layer Temperature Profile")
    ax.set_ylabel("Atmospheric Layer")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylim(-TOL, num_layers + TOL)

    plt.show()


#    ╔═╗     ╔═╗             ╔═╗ ╔═╗       ╔═╗               ╔═╗     ╔═╗
#    ║ ║     ║ ║ ╔═══════╗   ║ ║ ╚═╝ ╔═════╝ ║ ╔═══════╗   ╔═╝ ╚═╗   ╚═╝ ╔═══════╗ ╔═══════╗
#    ║ ║     ║ ║ ║ ╔═══╗ ║   ║ ║ ╔═╗ ║ ╔═══╗ ║ ║ ╔═══╗ ║   ╚═╗ ╔═╝   ╔═╗ ║ ╔═══╗ ║ ║ ╔═══╗ ║
#    ║ ╚═╗ ╔═╝ ║ ║ ║   ║ ║   ║ ║ ║ ║ ║ ║   ║ ║ ║ ║   ║ ║     ║ ║     ║ ║ ║ ║   ║ ║ ║ ║   ║ ║
#    ╚═╗ ╚═╝ ╔═╝ ║ ╚═══╝ ╚═╗ ║ ║ ║ ║ ║ ╚═══╝ ║ ║ ╚═══╝ ╚═╗   ║ ╚═══╗ ║ ║ ║ ╚═══╝ ║ ║ ║   ║ ║
#      ╚═════╝   ╚═════════╝ ╚═╝ ╚═╝ ╚═══════╝ ╚═════════╝   ╚═════╝ ╚═╝ ╚═══════╝ ╚═╝   ╚═╝
#
#DEBUG = True
#COEF_DEBUG = True
#create_coef_matrix(2, 1)
#create_coef_matrix(2, 0)
#model_atmosphere(1, S_0, e = 1, a=a)
#model_atmosphere(1, S_0, e = 0.5, a=a)
#model_atmosphere(1, S_0, e = 0, a=a)
#plot_temp_profile(3.4, 0, 0, 0)
#plot_temp_profile(1, 1, 1, 1, title = 0)


#    ╔═════════╗                       ╔═╗     ╔═╗
#    ╚═══╗ ╔═══╝   ╔═══╗     ╔═════╗ ╔═╝ ╚═╗   ╚═╝ ╔═══════╗ ╔═══════╗
#        ║ ║     ╔═╝   ╚═╗ ╔═╝   ╔═╝ ╚═╗ ╔═╝   ╔═╗ ║ ╔═══╗ ║ ║ ╔═══╗ ║
#        ║ ║     ║ ╔═════╝ ╚═══╗ ╚═╗   ║ ║     ║ ║ ║ ║   ║ ║ ║ ║   ║ ║
#        ║ ║     ║ ╚═════╗ ╔═══╝ ╔═╝   ║ ╚═══╗ ║ ║ ║ ║   ║ ║ ║ ╚═══╝ ║
#        ╚═╝     ╚═══════╝ ╚═════╝     ╚═════╝ ╚═╝ ╚═╝   ╚═╝ ╚═════╗ ║
#                                                              ╔═══╝ ║
#                                                              ╚═════╝

#Plot the 1-layer atmosphere for the first part of question 3
experiment1()

#Plot Earth's Surface Temp as function of num-layers
plot_nlayer_surftemp(10, S_0, e = 0.255, albedo = A, temp_marker = 288)

#Show Earth's Temp profile for 4-layer
plot_temp_profile(4, S_0, e = 0.255, albedo = A, temp_marker = 288)

#Show Earth's Temp profile for 5-layer
plot_temp_profile(5, S_0, e = 0.255, albedo = A, temp_marker = 288)


#find difference between 4 and 5 layer surface temp)
temp_4layer = model_atmosphere(4, S_0, e = 0.255, albedo = A)[0]
temp_5layer = model_atmosphere(5, S_0, e = 0.255, albedo = A)[0]
#Print the difference to make a choice on which layer to use
print(f"4-Layer Difference from 288K:\t {np.abs(288 - temp_4layer)}")
print(f"5-Layer Difference from 288K:\t {np.abs(288 - temp_5layer)}")
print(f"Temperature Difference:\t {temp_5layer - temp_4layer}")

#Check out Venus's surface temperature based on N-Layers
plot_nlayer_surftemp(num_layers = 120,
                     solar_irradiance = 2600,
                     e = 1,
                     albedo = 0.77, #Bond albedo from NASA fact sheet
                     temp_marker= 700,
                     title = "N-Layer Venus Atmosphere"
                    )#

#go for large scale nuclear war
plot_temp_profile(num_layers = 5,
                  solar_irradiance = S_0,
                  e = 0.5,
                  albedo = 0, #completely absorb the sw
                  title = "Nuclear Winter Temperature Profile",
                  sw_absorbing_layer = 5 #The final atmosphere layer absorbs it all!
)
