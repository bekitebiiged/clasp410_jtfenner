#Import modules and such
from scipy.integrate import ode, RK45
import numpy as np
import matplotlib.pyplot as plt



#Set the plotting styles:
#Update these to adjust plotting aesthetics
plt.style.use("Solarize_Light2")
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.facecolor'] = 'gainsboro'
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['legend.facecolor'] = 'oldlace'


#Define the methods to be used later
def lotvolt_competition(t, N, coefs):
    '''
    Calculates Lotka-Volterra population change rates for two species
    with populations N at time t
    ===============
        Inputs
    ===============
        t: float
            time (in years)
        N: list of floats [pop1, pop2]
            population density vector [pop1, pop2] [0,1]
        coefs: list of floats [a, b, c, d]
            values of a, b, c, d respectively.
            a, c:
            b, d:
    ===============
        Returns
    ===============
        dNdT: nparray of floats
            change rates [pop1 rate, pop2 rate], density per year
    ====================
        Example Usage
    ====================
    curr_pop_change = lotvolt_predprey(t = 1,                   # time in years
                                        N = curr_population,    # [curr_pop1_density, curr_pop2_density]
                                        coefs = [1, 2, 3, 4],   # [a, b, c, d]
                                    )
    '''
    #create empty vector to hold calulated rates
    dNdT = np.zeros(2)

    a = coefs[0]
    b = coefs[1]
    c = coefs[2]
    d = coefs[3]
    #Calculate RHS of Lotka Volterra Model
    dNdT[0] = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dNdT[1] = c*N[1]*(1-N[1]) - d*N[0]*N[1]

    return dNdT

def lotvolt_predprey(t, N, coefs):
    '''
    Calculates Predator-Prey Lotka-Volterra population change rates for two species
    with populations N at time t
    ===============
        Inputs
    ===============
        t: float
            Time (in years)
        N: list of floats [prey_density, pred_density]
            Population density vector [prey, predator] [0,1]
        coefs: list of floats [a, b, c, d]
            Values of a, b, c, d respectively.
            a:
            b:
            c:
            d:
    ===============
        Returns
    ===============
        dNdT: nparray [floats]
            change rates [prey rate, pred rate], density per year
    ====================
        Example Usage
    ====================
    curr_pop_change = lotvolt_predprey(t = 1,                   # time in years
                                        N = curr_population,    # [curr_pred_density, curr_prey_density]
                                        coefs = [1, 2, 3, 4],   # [a, b, c, d]
                                    )
    '''
    #create empty vector to hold calulated rates
    dNdT = np.zeros(2)
    # set a,b,c,d values for readability
    a = coefs[0]
    b = coefs[1]
    c = coefs[2]
    d = coefs[3]

    #Calculate RHS of Lotka Volterra Model
    dNdT[0] = (a*N[0]) - (b*N[0]*N[1])
    dNdT[1] = (-c*N[1]) + (d*N[0]*N[1])


    return dNdT

def get_lotvolt_comp_solution(coefs):
    '''
    Calculates the equilibrium state solution for the species-competition LotVolt Computation solution
    ===============
        Inputs
    ===============

        coefs: list of floats [a, b, c, d]
            Values of a, b, c, d respectively.
            a, c:
                Reproduction rates of pops 1, 2 respectively
            b, d:
                scales impact of species on each other (2 on 1, 1 on 2)
    ===============
        Returns
    ===============
        [N1, N2]: floats
            Final steady state for N1 and N2
    ====================
        Example Usage
    ====================
        solution_pops = get_lotvolt_comp_solution([1, 2, 3, 4]) # [a, b, c, d]

    '''
    #https://docs.python.org/3/tutorial/errors.html
    #Try to get N1 and N2, making sure to let the user know if the denom is 0
    N1 = None
    N2 = None
    # set a,b,c,d values for readability
    a = coefs[0]
    b = coefs[1]
    c = coefs[2]
    d = coefs[3]

    #Try to calculate N1,N2, return an error for the divide by zero
    try:
        N1 = c*(a - b) / (c*a - b*d)
        N2 = a*(c - d) / (c*a - b*d)
    except ZeroDivisionError:
        print("Dividing by zero!! No solution because c*d = a*d")

    return np.abs([N1,N2]) #added absolute values because sometimes returned -0

def euler_solve_compmodel(N, coefs, d_t, timelength):
    '''
    Use Euler's method to compute a numerical solution of the Lotka-Volt Competition Model
    ===============
        Inputs
    ===============
        N: list of floats [pop1, pop2]
            List of initial population densities for two species in competition
        coefs: list of floats [a, b, c, d]
            Values of a, b, c, d respectively.
            a, c:
                Reproduction rates of pops 1, 2 respectively
            b, d:
                scales impact of species on each other (2 on 1, 1 on 2)
        d_t:
            timestep size (same units as timelength)
        timelength:
            period of time to run the model
    ===============
        Returns
    ===============
        (N1, N2): touple of float lists
            List of population densities over time, each index is a timestep of d_t
        timeline: list of floats
            List of times simulated (to aid in graphing)
    ====================
        Example Usage
    ====================
    populations_over_time, timeline = euler_solve_compmodel(N = [0.3, 0.6],     # [pop1, pop2]
                                                            coefs = [1,2,3,4],  # [a, b, c, d]
                                                            d_t = 1,            # year
                                                            timelength = 100)   # years
    '''
    #define the first N1 and N2 values
    curr_N1 = N[0]
    curr_N2 = N[1]

    #store the first N1 and N2 values into the list of all population densities
    N1 = [curr_N1]
    N2 = [curr_N2]

    #set the first timestep as T = 0
    timeline = [0]

    #compute number of steps (ensure no divide by zero!)
    try:
        num_steps = int(timelength / d_t) #units in timesteps
    except ZeroDivisionError:
        print("d_t must be greater than 0!")

    #loop over num_steps (exclusive to timelength i.e. [0, timelength) )
    for timestep in range(1,num_steps+1):
        #calculate the change in population for next time step
        dN_dT = lotvolt_competition(timestep, [curr_N1, curr_N2], coefs)
        #change the populations according to the calc rates
        curr_N1 += dN_dT[0]
        curr_N2 += dN_dT[1]
        #append these new population numbers to the list
        N1.append(curr_N1)
        N2.append(curr_N2)
        #add the current time step to the timeline units of d_t (years)
        timeline.append(timestep * d_t)

    return (N1, N2), timeline

def euler_solve_predmodel(N, coefs, d_t, timelength):
    '''
    Use Euler's method to compute a numerical solution of the Lotka-Volt Predator-Prey Model
    ===============
        Inputs
    ===============
        N: list of floats [prey, pred]
            List of initial population densities for prey and predator respectively
        coefs: list of floats [a, b, c, d]
            Values of a, b, c, d respectively.
            a, c:
                Reproduction rates of pops 1, 2 respectively
            b, d:
                scales impact of species on each other (2 on 1, 1 on 2)
        d_t:
            timestep size (same units as timelength)
        timelength:
            period of time to run the model
    ===============
        Returns
    ===============
        (N1, N2): touple of float lists
            List of population densities over time, each index is a timestep of d_t
        timeline: list of floats
            List of times simulated (to aid in graphing)

    ====================
        Example Usage
    ====================
    populations_over_time, timeline = euler_solve_compmodel(N = [0.3, 0.6],     # [prey, pred]
                                                            coefs = [1,2,3,4],  # [a, b, c, d]
                                                            d_t = 1,            # year
                                                            timelength = 100)   # years
    '''
    #define the first N1 and N2 values
    curr_N1 = N[0]
    curr_N2 = N[1]

    #store the first N1 and N2 values into the list of all population densities
    N1 = [curr_N1]
    N2 = [curr_N2]

    #set the first timestep as T = 0
    timeline = [0]

    #compute number of steps (ensure no divide by zero!)
    num_steps : int
    try:
        num_steps = int(timelength / d_t) #units in timesteps
    except ZeroDivisionError:
        print("d_t must be greater than 0!")

    #loop over num_steps (exclusive to timelength i.e. [0, timelength) )
    for timestep in range(1,num_steps+1):
        #calculate the change in population for next time step
        dN_dT = lotvolt_predprey(timestep, [curr_N1, curr_N2], coefs)
        #change the populations according to the calc rates
        curr_N1 += dN_dT[0]
        curr_N2 += dN_dT[1]
        #append these new population numbers to the list
        N1.append(curr_N1)
        N2.append(curr_N2)
        #add the current time step to the timeline units of d_t (years)
        timeline.append(timestep / (1 / d_t))

    return (N1, N2), timeline

def rk45_solve_compmodel(N, coefs, d_t, timelength):
    '''
    Function to find a solution to the competition Lotka-Volt Model using SciPy integrator
    ===============
        Inputs
    ===============
        N: list of floats [pop1, pop2]
            List of initial population densities for pop1 and pop2 respectively
        coefs: list of floats [a, b, c, d]
            Values of a, b, c, d respectively.
            a, c:
                Reproduction rates of pops 1, 2 respectively
            b, d:
                scales impact of species on each other (2 on 1, 1 on 2)
        d_t:
            timestep size (same units as timelength)
        timelength:
            period of time to run the model
    ===============
        Returns
    ===============
        (N1, N2): touple of float lists
            List of population densities over time, each index is a timestep of d_t
        timeline: list of floats
            List of times simulated (to aid in graphing)
    ====================
        Example Usage
    ====================
    populations_over_time, timeline = rk45_solve_compmodel(N = [0.3, 0.6],     # [pop1, pop2]
                                                            coefs = [1,2,3,4],  # [a, b, c, d]
                                                            d_t = 1,            # year
                                                            timelength = 100)   # years
    '''
    #Create a integrator object
    r = ode(lotvolt_competition).set_integrator('dopri5')
    # Set the initial state of the function with the time = 0, population densities, and coefficient values
    r.set_initial_value(N, 0).set_f_params(coefs)
    #set output
    n1 = []
    n2 = []
    #continue to timestep the function while it is able to and within the timelength
    while r.successful() and r.t < timelength:
        r.integrate(r.t + d_t)
        n1.append(r.y[0])
        n2.append(r.y[1])

    return [n1, n2]

def rk45_solve_predmodel(N, coefs, d_t, timelength):
    '''
    Function to find a solution to the competition Lotka-Volt Model using SciPy integrator
    ===============
        Inputs
    ===============
        N: list of floats [prey, pred]
            List of initial population densities for prey and predator respectively
        coefs: list of floats [a, b, c, d]
            Values of a, b, c, d respectively.
            a, c:
                Reproduction rates of pops 1, 2 respectively
            b, d:
                scales impact of species on each other (2 on 1, 1 on 2)
        d_t:
            timestep size (same units as timelength)
        timelength:
            period of time to run the model
    ===============
        Returns
    ===============
        (N1, N2): touple of float lists
            List of population densities over time, each index is a timestep of d_t
        timeline: list of floats
            List of times simulated (to aid in graphing)
    ====================
        Example Usage
    ====================
    populations_over_time, timeline = rk45_solve_predmodel(N = [0.3, 0.6],     # [prey, pred]
                                                            coefs = [1,2,3,4],  # [a, b, c, d]
                                                            d_t = 1,            # year
                                                            timelength = 100)   # years
    '''

    #Create a integrator object
    r = ode(lotvolt_predprey).set_integrator('dopri5')
    # Set the initial state of the function with the time = 0, population densities, and coefficient values
    r.set_initial_value(N, 0).set_f_params(coefs)
    #set output
    n1 = []
    n2 = []
    #continue to timestep the function while it is able to and within the timelength
    while r.successful() and r.t < timelength:
        r.integrate(r.t + d_t)
        n1.append(r.y[0])
        n2.append(r.y[1])

    return [n1, n2]

def run_question1_compmodel():
    #completes question 1 for the CLIMATE410 Homework

    #Adjust the initial conditions here
    init_pops = [0.3, 0.6] #[species1, species2]
    coefs = [1, 2, 1, 3] #[a, b, c, d]
    d_t = 1 #years
    timelength = 100 #years

    #get the calculations
    euler_pops, timeline = euler_solve_compmodel(init_pops, coefs, d_t, timelength)
    rk45_pops = rk45_solve_compmodel(init_pops, coefs, d_t, timelength)
    steady_state_solution = get_lotvolt_comp_solution(coefs)

    #Create the fig and axis objects. They should sharex and y to be able to compare easily
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 6), sharex=True, sharey=True)

    #Plot the steady state solutions in black
    ax1.hlines(steady_state_solution, 0, timelength, ls = 'solid',
            linewidth = 2, color = 'black', label = 'Steady Soltution')
    ax2.hlines(steady_state_solution, 0, timelength, ls = 'solid',
            linewidth = 2, color = 'black', label = 'Steady Soltution')

    ax1.plot(timeline, euler_pops[0], color = 'teal', alpha = 0.8,  label = "N1")
    ax1.plot(timeline, euler_pops[1], color = 'sandybrown', alpha = 0.8, ls = '--', label = "N2")
    ax2.plot(rk45_pops[0], color = 'teal', alpha = 0.8,  label = "N1")
    ax2.plot(rk45_pops[1], color = 'sandybrown', alpha = 0.8, ls = '--', label = "N2")



    fig.supxlabel("Time (years)")
    ax1.set_ylabel("Population Density")
    ax2.set_ylabel("Population Density")
    #Turn the yaxis ticks back on
    ax2.yaxis.set_tick_params(labelbottom=True)

    fig.suptitle("Lokta-Volterra Competition Model Solutions"
                + f"\n $\\Delta{{T}}$ = {d_t} years")
    ax1.set_title("Euler")
    ax2.set_title("RK45")
    ax1.legend()
    ax2.legend()
    plt.show()

def run_question1_predmodel():
    #completes question 1 for the CLIMATE410 Homework

    #Adjust the initial conditions here
    init_pops = [0.3, 0.6] #[species1, species2]
    coefs = [1, 2, 1, 3] #[a, b, c, d]
    d_t = 0.05 #years
    timelength = 100 #years
    euler_pops, timeline = euler_solve_predmodel(init_pops, coefs, d_t, timelength)
    rk45_pops = rk45_solve_predmodel(init_pops, coefs, d_t, timelength)
    #steady_state_solution = get_lotvolt_comp_solution(coefs)
    #Create the fig and axis objects. They should sharex and y to be able to compare easily
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 6), sharex=True, sharey=True)
    ax1.plot(euler_pops[0], color = 'teal', alpha = 0.8,  label = "Prey")
    ax1.plot(euler_pops[1], color = 'sandybrown', alpha = 0.8, ls = '--', label = "Predator")
    ax2.plot(timeline, rk45_pops[0], color = 'teal', alpha = 0.8,  label = "Prey")
    ax2.plot(timeline, rk45_pops[1], color = 'sandybrown', alpha = 0.8, ls = '--', label = "Predator")

    fig.supxlabel("Time (years)")
    ax1.set_ylabel("Population Density")
    ax2.set_ylabel("Population Density")
    #Turn the yaxis ticks back on
    ax2.yaxis.set_tick_params(labelbottom=True)
    # Set the ylim to 0, 1 to ensure physical values
    ax1.set_ylim(0, 1)

    fig.suptitle("Lokta-Volterra Pred-Prey Model Solutions"
                + f"\n $\\Delta{{T}}$ = {d_t} years")
    ax1.set_title("Euler")
    ax2.set_title("RK45")
    ax1.legend()
    ax2.legend()
    plt.show()


#Uncomment print statements to test the algebraic solution function
#print(get_lotvolt_comp_solution(1, 1, 1, 1)) #returns divide by 0 error (good)
#print(get_lotvolt_comp_solution(2, 1, 4, 4)) #return [1, 0] (good)
#print(get_lotvolt_comp_solution(4,4,2,5)) #return [-0, 1] (good enough)
#print(get_lotvolt_comp_solution(3, 4, 5, 6)) #good

#Run the functions to complete each question
run_question1_compmodel()
run_question1_predmodel()