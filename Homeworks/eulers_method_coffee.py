import numpy as np
import matplotlib.pyplot as plt

plt.style.use('Solarize_Light2')
#Set constant value
K = 0.1
#Set the room temperature
ENVIORNMENT_TEMP = 18 #degrees C
#Define the initial temperature
INIT_T = 145 #degrees C

def model_cooling(init_temp, d_t, environ_temp, k = 0.1, tol = 0.1):
    '''
    Approximates cooling time based on Newton's Law of Cooling via Euler's Method (numerical estimation)

    ===============
        INPUTS
    ===============
        init_temp:
            initial temperature from which to cool.
        d_t:
            timestep to use for estimation
        environ_temp:
            temperature of the enviornement (to which the material will cool)
        k:
            constant K
        tol:
            difference from environment temperature
    ================
        RETURNS
    ================
        temperatures:
            list of temperatures calculated temperatures[-1] give final temp
        time_taken:
            calculated time taken
    '''

    #create a current temperature which is the initial temperature
    curr_T = init_temp
    #create a list of temperatures with the first entry being the init_temp/current curr_temp
    temperatures = [curr_T]

    time_taken = 0

    #loop while the absolute difference between the current temperature and enviornment temp
    # is greater than the tolerance
    while( np.abs( curr_T - environ_temp) > tol ):
        #calculate the next timestep using newton's law of cooling
        next_T = curr_T - ( k * d_t * (curr_T - environ_temp) )
        #append the next temperature to the overall temp list
        temperatures.append(next_T)
        #set the current temperature as the calculated temp value
        curr_T = next_T
        time_taken += d_t


    return temperatures, time_taken


def calculate_temp_time(init_temp, final_temp, env_temp = 20, k = 1):
    '''
    INPUTS
    -----------------
    init_temp:
        Initial Temperature (Celcius), Starting temperature from which to cool
    final_temp:
        Final Temperature (Celcius), Temperature to cool to
    env_temp: Defaults to 20 (Celcius)
        Temperature of the environment/air
    k:  Defaults to 20 (Celcius)
        Constant for cooling

    RETURNS
    -----------------
    total_time: #Units in s^-1
        The calculated time using
        t = -(1/k) * ln( (env_temp - env_temp) / (init_temp - env_temp) )
    '''
    #Calcalate the total time using the solution in the docstring
    total_time = -(1 / k) * np.log( (final_temp - env_temp) / (init_temp - env_temp) )

    return total_time


#create list of temperatures (temperature lists)
temperatures = []
#create dictionary of cooling times calculated for each timestep
cooling_times = {'euler' : [], 'analytic': []}
#creat list for timesteps
timesteps = []
#loop through timesteps
for d_t in np.arange(9,0.01, -0.1):

    #calculate the numeric and analytic solutions
    test, test_time = model_cooling(INIT_T, d_t = d_t,environ_temp = ENVIORNMENT_TEMP, k = K, tol = 0.1)
    verify_test_time = calculate_temp_time(INIT_T, final_temp = test[-1], env_temp= ENVIORNMENT_TEMP, k = K)

    #add cooling times for this timestep onto the cooling dicts
    cooling_times['euler'].append(test_time)
    cooling_times['analytic'].append(verify_test_time)
    #save the whole cooling timeseries  for this d_t
    temperatures.append(test)
    #append timestep to list
    timesteps.append(d_t)


#Create fig, ax to plot results
fig, ax = plt.subplots(1, 1, figsize = (8, 6))

#plot euler's method
ax.plot(np.log(timesteps), cooling_times['euler'], label = "Euler's Method")
#make the line be dashed for accessbility


ax.plot(np.log(timesteps), cooling_times['analytic'], ls='-.', label = "Analytic")

ax.set_title("Cooling Time Calcuated (Numerical Method vs Analytic Solution)\n"
             + "145C $\\to$ 18C, K = 0.1")

ax.xaxis.set_inverted(True)
ax.set_xlabel("Timestep ($\\ln{\\Delta \\text{t}}$(s))")
ax.set_ylabel("Cooling Time ($\\text{s}$)")
ax.legend()

plt.show()



