import numpy as np
#Set constant value
k = 0.1
#Set the room temperature
T_s = 18 #degrees C
#Set a delta_t (timestep
d_t = 2 #seconds
#Define the initial temperature
init_T = 145 #degrees C
#set a curr temp as the inital temperature
curr_T = init_T
#Define a tolerance
tol = 0.2 # tolerance in Celcius/Kelvin
#create list to store temperature values
temperatures = [init_T]

while( np.abs( curr_T - T_s) > tol ):
    #print(f"Curr Temp:\t{curr_T}")
    next_T = curr_T - ( k * d_t * (curr_T - T_s) )
    #print(f"Next Temp:\t{next_T}")
    temperatures.append(next_T)
    curr_T = next_T


print(f"Final Temp:\t{curr_T}")
print(f"Time Taken:\t{len(temperatures)*d_t}")
print(f"Temperatures:\t{temperatures}")
