'''






'''
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.animation as animation
import matplotlib.lines as lines
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

S_0 = 1370 # Wm^-2
THERMAL_DIFFUSIVITY = 100 # m^2s^-1
A = 6378e3 #m
OCEAN_DEPTH = 50 # m
STEFAN_BOLTZMANN = 5.67e-8 # J m^-2 s^-1 K^-4
EMISSIVITY = 1.0 # unitless

C_SEAWATER = 4.2e6 # J kg^-1 K^-1
RHO_SEAWATER = 1020 # kg m^-3

ALBEDO_ICE = 0.6
ALEBDO_GROUND = 0.3
SECONDS_PER_YEAR = 1 * 365 * 24 * 60 * 60 #seconds per year
STEADY_STATE_TIME = 20000 #10000 years to reach a steady state

npoints = 18
yspacing = 180 / npoints
lats = np.linspace(yspacing/2., 180-yspacing/2, npoints) # Lat cell centers
areas = np.pi * ( (A + 50)**2 - A**2) * np.sin(np.deg2rad(lats)) #Areas for each latband
# Set up plotting defaults
lat_ticks = np.linspace(-90, 90, num = npoints, endpoint=True)
xlabel = "Latitude ($^{\circ}$)"
ylabel = "Temperature ($C^{\circ}$)"

'''
Helpful functions for the Snowball Earth lab.
'''
def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required.
        0 corresponds to the south pole, 180 to the north.

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''

    # Set initial temperature curve
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])
    # Get base grid:
    npoints = T_warm.size
    dlat = 180 / npoints  # Latitude spacing.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints)  # Lat cell centers.
    # Fit a parabola to the above values
    coeffs = np.polyfit(lats, T_warm, 2)
    # Now, return fitting sampled at "lats".
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp

def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees
    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)
    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)
    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation

'''

Other Functions


'''

def time_step(temp_vector, area_vector, insolation_vector, d_time, y_spacing, therm_diff = 100,
              emissivity = EMISSIVITY, albedo = None,
              basic_diffusion = False, diffusion_with_correction = False, full_calculation = True):
    '''
    Function to calculate the next timestep for diffusive heat spread
    '''
    #def ghost_nodes(vector):
    #   return np.concatenate(([vector[0]],vector,[vector[-1]]), axis = 0)
    #temp_vector = ghost_nodes(temp_vector)
    #area_vector = ghost_nodes(area_vector)
    #insolation_vector = ghost_nodes(insolation_vector)

    dy = ( ( A * np.pi * y_spacing ) / 180 ) #dy in meters (y_spacing is in degrees)
    N = len(temp_vector)

    # Create B matrix:
    B = np.diag(np.zeros(N), k = 0)
    B += np.diag(np.ones(N-1), k = -1) * -1
    B += np.diag(np.ones(N-1), k = 1)
    B[0, :] = B[-1, :] = 0
    # Create K matrix:
    K = np.diag(np.ones(N), k = 0) * -2
    K += np.diag(np.ones(N - 1), k = 1)
    K += np.diag(np.ones(N - 1), k = -1)
    K[0,1] = K[-1,-2] = 2
    K = K / (dy**2)

    #Create L matrix
    L = np.identity(N) - therm_diff*d_time*K

    # calculate the radiative forcing term
    if DEBUG:
        print(f"K is : {K}")
        print(f"B is : {B}")
        print("K, L created")
        print(f"{temp_vector}")

    factor = (therm_diff * d_time * B.dot(temp_vector) * B.dot(area_vector)) / (area_vector * 4 * (dy**2) )

    if DEBUG:
        print(f"Spherical Correction: {factor}")
    # calculate the radiative forcing factor
    albedo_vector = np.ones(N)
    if albedo is None:
        ice_locs = ( temp_vector <= -10 )
        albedo_vector[ice_locs] = ALBEDO_ICE
        albedo_vector[~ice_locs] = ALEBDO_GROUND

    else:
        albedo_vector = albedo_vector * albedo
    #print(f"ALBEDO VECTOR: {albedo_vector}")
    radiative_forcing = (np.multiply(insolation_vector, (1 - albedo_vector) )
                        - ( emissivity * STEFAN_BOLTZMANN * np.power( (temp_vector + 273.15) , 4) )
                        )
    radiative_forcing = (d_time / (RHO_SEAWATER * C_SEAWATER * OCEAN_DEPTH) ) * radiative_forcing


    # declare the combined vector
    combined_vector : object
    # calculate the combined vector based on user choice
    if basic_diffusion:
        combined_vector = temp_vector
    elif diffusion_with_correction:
        combined_vector = temp_vector + factor
    elif full_calculation:
        combined_vector = temp_vector + factor + radiative_forcing

    next_step = np.linalg.solve(L, combined_vector)

    return next_step

def calc_steady_state(init_conds, area_vector, insolation_vector, d_time = SECONDS_PER_YEAR, y_spacing = yspacing, therm_diff = 100,
                        emissivity = EMISSIVITY, albedo = None):

    state = time_step(init_conds, area_vector, insolation_vector, d_time,
                    y_spacing = y_spacing, therm_diff= therm_diff,
                    emissivity= emissivity, albedo = albedo)
    for t in range(0, STEADY_STATE_TIME, 1):
        state = time_step(state, area_vector, insolation_vector, d_time,
                        y_spacing = y_spacing, therm_diff= therm_diff,
                        emissivity= emissivity, albedo = albedo)
    return state

def plot_steady_state(lats, temps, reference_line = None,
                        title = "Steady State Temperatures", curve_label = "Example",
                        xlabel = xlabel, ylabel = ylabel):
    '''

    '''
    fig, ax = plt.subplots(1, 1, figsize = (6, 4))
    curve, = ax.plot(lats, temps, label = curve_label)
    if reference_line is not None:
        if isinstance(reference_line, Iterable):
            for curr_line in reference_line:
                x, y = curr_line.get_data()
                ax.plot(x, y, label = curr_line.get_label(), alpha = 0.8)
        else:
            x, y = reference_line.get_data()
            ax.plot(x, y, label = reference_line.get_label(), alpha = 0.8)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    return curve

def validation_test():
    basic_diff = time_step(warm_earth, areas, insol, d_time = SECONDS_PER_YEAR, albedo = 0.3,
                        y_spacing = yspacing, therm_diff= THERMAL_DIFFUSIVITY,
                        basic_diffusion=True)

    diffusion_with_corr = time_step(warm_earth, areas, insol, d_time = SECONDS_PER_YEAR, albedo = 0.3,
                        y_spacing = yspacing, therm_diff= THERMAL_DIFFUSIVITY,
                        diffusion_with_correction=True)

    diff_corr_radiative = time_step(warm_earth, areas, insol, d_time = SECONDS_PER_YEAR, albedo = 0.3,
                            y_spacing = yspacing, therm_diff = THERMAL_DIFFUSIVITY,
                            full_calculation=True)

    for t in range(0, 10000, 1):
        basic_diff = time_step(basic_diff, areas, insol, d_time = SECONDS_PER_YEAR, albedo = 0.3,
                        y_spacing = yspacing, therm_diff= THERMAL_DIFFUSIVITY,
                        basic_diffusion=True)

        diffusion_with_corr = time_step(diffusion_with_corr, areas, insol, d_time = SECONDS_PER_YEAR, albedo = 0.3,
                        y_spacing = yspacing, therm_diff= THERMAL_DIFFUSIVITY,
                        diffusion_with_correction=True)

        diff_corr_radiative = time_step(diff_corr_radiative, areas, insol, d_time = SECONDS_PER_YEAR, albedo = 0.3,
                                y_spacing = yspacing, therm_diff = THERMAL_DIFFUSIVITY,
                                full_calculation=True)

    fig, ax = plt.subplots(1, 1, figsize = (6, 4))

    ax.plot(lats, warm_earth, label = 'initial conditions')
    ax.plot(lats, basic_diff, label = 'basic diffusion')
    ax.plot(lats, diffusion_with_corr, label = 'diff + spherical correction')
    ax.plot(lats, diff_corr_radiative, label = 'diff + sphcorr + radiation')
    ax.legend()
    ax.set_title("Steady-State Temperatures")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


warm_earth = temp_warm(lats)
insol = insolation(S_0, lats)

#validation_test()

#Tune model to reproduce warm-Earth equilibrium

#Loop from 0 - 150 [THERMAL DIFFUSIVITY]
variable_diffs_steady_state = []
therm_diffs = np.linspace(0, 150, num = 150, endpoint=True)
for therm_diff in therm_diffs:
    curr_step = time_step(warm_earth, areas, insol, d_time = SECONDS_PER_YEAR, albedo = 0.3,
                        y_spacing = yspacing, therm_diff = therm_diff,
                        full_calculation=True)
    for t in range(0, 10000, 1):
        curr_step = time_step(curr_step, areas, insol, d_time = SECONDS_PER_YEAR, albedo = 0.3,
                        y_spacing = yspacing, therm_diff = therm_diff,
                        full_calculation=True)

    variable_diffs_steady_state.append(curr_step)

fig, ax = plt.subplots(1, 1, figsize = (6, 4))
warm_earth_line, = ax.plot(lat_ticks, warm_earth, label = ('Warm Earth'), alpha = 0.5)
line, = ax.plot(lat_ticks, variable_diffs_steady_state[0], label = ('$\lambda$' + f'{0}'))
ax.set_title("Steady State Temperatures \n Variable $\lambda$")
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
def update(frame):
    y = variable_diffs_steady_state[frame]
    line.set_data(lat_ticks, y)
    line.set_label('$\lambda$ = ' + f"{frame}" + " $m^{2}s^{-1}$")
    ax.set_ylim(np.min(warm_earth) - 10, np.max([y, warm_earth])+10)
    ax.legend(loc = 'upper right')
    return line,

# Set up formatting for the movie files
#https://stackoverflow.com/questions/31192125/saving-animations-using-ffmpeg-and-matplotlib
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani = animation.FuncAnimation(fig = fig, func = update, frames = 150, interval = 50, repeat = True)
ani.save('Labs/Lab4_SnowballEarth/figures/ThermalDiffAnimation.mp4', writer = writer)
plt.show()
#Loop from 0 - 1 [EMISSIVITY]
variable_emissivity_steady_state = []
emissivities = np.linspace(0, 1, endpoint=True)
for emissivity in emissivities:
    curr_step = time_step(warm_earth, areas, insol, d_time = SECONDS_PER_YEAR, albedo = 0.3,
                        y_spacing = yspacing, therm_diff = THERMAL_DIFFUSIVITY,
                        emissivity = emissivity,
                        full_calculation=True)
    for t in range(0, 10000, 1):
        curr_step = time_step(curr_step, areas, insol, d_time = SECONDS_PER_YEAR, albedo = 0.3,
                        y_spacing = yspacing, therm_diff = THERMAL_DIFFUSIVITY,
                        emissivity= emissivity,
                        full_calculation=True)

    variable_emissivity_steady_state.append(curr_step)
fig, ax = plt.subplots(1, 1, figsize = (6, 4))
ax.plot(lat_ticks, warm_earth, label = ('Warm Earth'), alpha = 0.5)
line, = ax.plot(lat_ticks, variable_emissivity_steady_state[0], label = ('$\epsilon$' + f'{0}'))

ax.set_title("Steady State Temperatures \n Variable Emissivity")
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
def update_emiss(frame):
    y = variable_emissivity_steady_state[frame]
    line.set_data(lat_ticks, y)
    line.set_label('$\epsilon$ = ' + f"{emissivities[frame]:.3f}")
    ax.legend()
    ax.set_ylim(np.min(warm_earth) - 10, np.max([y, warm_earth])+10)
    ax.legend(loc = 'upper right')
    return line,

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani = animation.FuncAnimation(fig = fig, func = update_emiss, frames = len(variable_emissivity_steady_state),
                               interval = 100, repeat = True)
ani.save('Labs/Lab4_SnowballEarth/figures/EmissivityAnimation.mp4', writer = writer)
plt.show()


#Question  2
warm_earth_line = lines.Line2D(lat_ticks, warm_earth, label = ('Warm Earth'), alpha = 0.5)
chosen_emiss = 0.735
chosen_therm_diff = 30
temp_title = ("Constructed Warm Earth Temperature\n $\epsilon =$ " + f"{chosen_emiss} | "
              + "$\lambda=$ " + f"{chosen_therm_diff} " + "$m^{2}s^{-1}$"
)
chosen_warm_earth_temps = calc_steady_state(warm_earth, areas, insol,
                                            therm_diff = chosen_therm_diff, emissivity= chosen_emiss, albedo = 0.3)
chosen_warm_earth_curve = plot_steady_state(lat_ticks, chosen_warm_earth_temps, reference_line=warm_earth_line, title = temp_title,
                                            curve_label= "Constructed Warm Earth")

#Question 3
hot_earth_init_temps = np.ones(lats.shape) * 60 #60C at all points
cold_earth_init_temps = hot_earth_init_temps.copy() * -1 #-60C at all points
hot_title = "Hot Earth Steady State"
cold_title = "Cold Earth Steady State"
hot_earth_temps = calc_steady_state(hot_earth_init_temps, areas, insol,
                                    therm_diff = chosen_therm_diff, emissivity= chosen_emiss)
hot_earth_curve = plot_steady_state(lat_ticks, hot_earth_temps,
                                    reference_line=chosen_warm_earth_curve, title = hot_title, curve_label= "Hot Earth")

cold_earth_temps = calc_steady_state(cold_earth_init_temps, areas, insol,
                                    therm_diff = chosen_therm_diff, emissivity= chosen_emiss)
cold_earth_curve = plot_steady_state(lat_ticks, cold_earth_temps,
                                    reference_line=chosen_warm_earth_curve, title = cold_title, curve_label= "Cold Earth")

flash_title = "Steady State Temperatures"
flash_frozen_temps = calc_steady_state(warm_earth, areas, insol, therm_diff= chosen_therm_diff, emissivity= chosen_emiss,
                                       albedo = 0.6)
flash_frozen_curve = plot_steady_state(lat_ticks, flash_frozen_temps, title = flash_title, reference_line= [hot_earth_curve, cold_earth_curve],
                                       curve_label = "Flash-Frozen Earth")

#Question 4
d_gamma = 0.05
avg_global_temps = []
gamma_list = np.arange(0.4, 1.4+d_gamma, d_gamma)
gamma_list = np.concat([gamma_list, sorted(gamma_list[:-1], reverse=True)], axis = 0)
curr_state_temps = cold_earth_init_temps.copy()
state_temps = [curr_state_temps]
for gamma in gamma_list:
    curr_insol = gamma * insolation(S_0, lats)
    curr_state_temps = calc_steady_state(curr_state_temps, areas, curr_insol,
                                        therm_diff= chosen_therm_diff, emissivity= chosen_emiss )
    #get the y-values from the curve (steady state temps)

    avg_global_temps.append(np.mean(curr_state_temps))
    state_temps.append(curr_state_temps)
fig, ax = plt.subplots(1, 1, figsize = (6, 4))
ax.plot(gamma_list, avg_global_temps)
#show direction arrows at multipler = 0.9 (about halfway through both sides)
quarter = len(gamma_list) // 4

ax.set_title("Average Global Temp vs Solar Multiplier")
ax.set_ylabel(ylabel)
ax.set_xlabel("Solar Multiplier ($\gamma$)")
ax.annotate('', xytext = (gamma_list[quarter], avg_global_temps[quarter]),
             xy= (gamma_list[quarter+1], avg_global_temps[quarter+1]),
             arrowprops = dict(shrink = 0.05)
)
ax.annotate('', xytext = (gamma_list[-quarter], avg_global_temps[-quarter]),
             xy= (gamma_list[(-quarter)+1], avg_global_temps[(-quarter)+1]),
             arrowprops = dict(shrink = 0.05),
)

ax.annotate('Initial State', xy = (gamma_list[0], avg_global_temps[0]+2),
            xytext = (gamma_list[0], avg_global_temps[0]+40),
            arrowprops= dict(arrowstyle = 'simple', color = 'orchid'),
            bbox = dict(facecolor = 'orchid', boxstyle='round,pad=0.5'), fontsize = 12 )
ax.plot(gamma_list[0], avg_global_temps[0], marker = '*', markersize = 8)

plt.show()

#for gamma, state_temp in zip(gamma_list, state_temps):
#    fig, ax = plt.subplots(1, 1, figsize = (6, 4))
#    ax.plot(lat_ticks, state_temp)
#    ax.set_title("Steady State Temp at $\gamma = $" + f"{gamma}")
#    plt.show()