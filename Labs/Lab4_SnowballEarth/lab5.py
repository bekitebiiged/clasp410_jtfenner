# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

'''
Helpful functions for the Snowball Earth lab.
'''
DEBUG = True

npoints = 18
S_0 = 1370 # Wm^-2
THERMAL_DIFFUSIVITY = 100 # m^2s^-1
A = 6378e3 #m
OCEAN_DEPTH = 50 # m
STEFAN_BOLTZMANN = 5.67e-8 # J m^-2 s^-1 K^-4
EMISSIVITY = 1.0 # unitless

C_SEAWATER = 4.2e6 # J m^-3 K^-1
RHO_SEAWATER = 1020 # kg m^-3

ALBEDO_ICE = 0.6
ALEBDO_GROUND = 0.3

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

def first_diff(vector, d_step = 1):
    '''
    Function to calculate first difference with the first and last boundary conditions
    at 0 [doesn't matter in this case]

    Works by creating ghost nodes at the top and bottom to find the central difference
    '''
    temp_vector = np.concatenate(([vector[0]],vector,[vector[-1]]), axis = 0)
    N = len(temp_vector)
    B = np.diag(np.ones(N - 1), k = -1) * -1
    B += np.diag(np.ones(N - 1), k = 1)
    B[-1, :] = B[0, :] = 0
    #print(B)
    #Make B include the 1 / 2*d_step
    #B = B / (2 * d_step)
    diff = B.dot(temp_vector)
    if DEBUG:
        print(f"Difference Vector: {diff}")
        print(f"OG Vector: {temp_vector}")
    #Calculate the first derivative
    return diff[1:-1]


def time_step(temp_vector, area_vector, insolation_vector, d_time, y_spacing, therm_diff = 100,
                basic_diffusion = False, diffusion_with_correction = False,
                full_calculation = True):
    '''
    Function to calculate the next timestep for diffusive heat spread
    '''
    N = len(temp_vector)
    K = np.diag(np.ones(N), k = 0) * -2
    K += np.diag(np.ones(N - 1), k = 1)
    K += np.diag(np.ones(N - 1), k = -1)

    #Add the boundary conditions
    K[0,1] = K[-1,-2] = 1
    K = K / (y_spacing**2)
    L = np.identity(N) - therm_diff*d_time*K
    # calculate the radiative forcing term
    if DEBUG:
        print(f"K is : {K}")
        print("K, L created")
        print(f"{temp_vector}")
        print(f"{first_diff(temp_vector, y_spacing)}")

    factor = therm_diff * d_time * first_diff(temp_vector, y_spacing) * first_diff(area_vector, y_spacing)

    if DEBUG:
        print(f"Spherical Correction: {factor}")
    # calculate the radiative forcing factor
    albedo_vector = np.zeros(N)
    ice_locs = ( temp_vector <= 10 )
    albedo_vector[ice_locs] = ALBEDO_ICE
    albedo_vector[~ice_locs] = ALEBDO_GROUND
    if DEBUG:
        print(f"ALBEDO VECTOR: {albedo_vector}")

    radiative_forcing = ( (d_time / ( RHO_SEAWATER * C_SEAWATER * OCEAN_DEPTH ))
                         * np.multiply(insolation_vector, (1-albedo_vector)
                          - EMISSIVITY * STEFAN_BOLTZMANN * (np.power(temp_vector, 4)))
    )

    # declare the combined vector
    combined_vector : object
    # calculate the combined vector based on user choice
    if basic_diffusion:
        combined_vector = temp_vector
    elif diffusion_with_correction:
        combined_vector = temp_vector + factor
    else:
        combined_vector = temp_vector + factor + radiative_forcing

    next_step = L.dot(combined_vector)

    return next_step

yspacing = 180 / npoints
lats = np.linspace(yspacing/2., 180-yspacing/2, npoints) # Lat cell centers
temps = temp_warm(lats)
init_cond = temp_warm(lats)
insol = insolation(S_0, lats)
areas = np.pi * ( (A + 50)**2 - A**2) * np.sin(np.deg2rad(lats))

basic_diff = time_step(init_cond, areas, insol, d_time = 1,
                       y_spacing = yspacing, therm_diff= THERMAL_DIFFUSIVITY,
                       diffusion_with_correction=True)

fig, ax = plt.subplots(1, 1, figsize = (8, 4))
ax.plot(lats, temps, label = 'initial conditions')
ax.plot(lats, basic_diff, label = 'basic diffusion')
ax.legend()
plt.show()
