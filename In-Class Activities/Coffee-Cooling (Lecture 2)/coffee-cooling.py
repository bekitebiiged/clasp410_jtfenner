import matplotlib.pyplot as plt
import numpy as np

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
        t = -(1/k) * ln( (final_temp - env_temp) / (init_temp - env_temp) )
    '''
    #Calcalate the total time using the solution in the docstring
    total_time = -(1 / k) * np.log( (final_temp - env_temp) / (init_temp - env_temp) ) 

    return total_time

    

ENV_TEMP = 20 #Celcius
K = 1 #s^-1

#FIRST EXPERIMENT WITHOUT CREAMER 
# 90C -> 60C
no_creamer_time = calculate_temp_time(90, 60)

#SECOND EXPERIMENT WITH CREAMER FIRST
# 85C -> 60C
creamer_first_time = calculate_temp_time(85, 60)

#THIRD EXPERIEMENT WITH CREAMER LAST
# 90C -> 65C
creamer_last_time = calculate_temp_time(90, 65) 


#REPORT THE FINAL TIMES FOR EACH
print(f"\t\t\tTIME TO COOL\n"
      f"\t90C to 60C Cooling Time/[NO\tCREAM\tADDED]:\t{no_creamer_time:5f} s\n"
      f"\t85C to 60C Cooling Time/[FIRST\tCREAM\tADDED]:\t{creamer_first_time:5f} s\n"
      f"\t90C to 65C Cooling Time/[LAST\tCREAM\tADDED]:\t{creamer_last_time:5f} s\n" 
)
