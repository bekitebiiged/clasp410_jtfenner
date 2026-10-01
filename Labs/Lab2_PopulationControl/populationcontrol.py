#Import modules and such
from scipy.integrate import ode, RK45
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("Solarize_Light2")

#Define methods to be used
def lotvolt_competition(t, N, coefs):
    '''
    Calculates Lotka-Volterra population change rates for two species
    with populations N at time t
    ===============
        Inputs
    ===============
        t: float
            time (in years)
        N: float
            population density vector [pop1, pop2] [0,1]
        a, c: float
            Reproduction rates of pops 1, 2 respectively
        b, d: float
            scales impact of species on each other (2 on 1, 1 on 2)
    ===============
        Returns
    ===============
        dNdT: nparray of floats
            change rates [pop1 rate, pop2 rate], density per year
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
        N: nparray [floats]
            Population density vector [prey, predator] [0,1]
        a: float
            Reproduction rates of prey
        b: float
            coefficient describing hunted-ness of prey by predator
        c: float
            Death rate of predators
        d: float
            Coefficient describing growth of predator due to prey availablity

    ===============
        Returns
    ===============
        dNdT: nparray [floats]
            change rates [prey rate, pred rate], density per year
    '''
    #create empty vector to hold calulated rates
    dNdT = np.zeros(2)

    a = coefs[0]
    b = coefs[1]
    c = coefs[2]
    d = coefs[3]

    #Calculate RHS of Lotka Volterra Model
    dNdT[0] = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dNdT[1] = -c*N[1]*(1-N[1]) + d*N[0]*N[1]
    return dNdT

def get_lotvolt_comp_solution(coefs):
    '''
    Calculates the equilibrium state solution for the species-competition LotVolt Computation solution

    ===============
        Inputs
    ===============


        a, c: float
            Reproduction rates of pops 1, 2 respectively
        b, d: float
            scales impact of species on each other (2 on 1, 1 on 2)

    ===============
        Returns
    ===============
        [N1, N2]: floats
            Final steady state for N1 and N2
    '''
    #https://docs.python.org/3/tutorial/errors.html
    #Try to get N1 and N2, making sure to let the user know if the denom is 0
    N1 = None
    N2 = None

    a = coefs[0]
    b = coefs[1]
    c = coefs[2]
    d = coefs[3]

    try:
        N1 = c*(a - b) / (c*a - b*d)
        N2 = a*(c - d) / (c*a - b*d)
    except ZeroDivisionError:
        print("Dividing by zero!! No solution because c*d = a*d")

    return np.abs([N1,N2]) #added absolute values because sometimes returned -0

def euler_solve_compmodel(pops, coefs, d_t, timelength):
    '''



    '''
    #create the euler solution function
    curr_N1 = pops[0]
    curr_N2 = pops[1]

    N1 = [curr_N1]
    N2 = [curr_N2]

    timeline = [0]
    num_steps = d_t * timelength # timesteps (years cancel out)

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

    return [N1, N2], timeline

def rk45_solve_compmodel(pop, coefs, d_t, timelength):
    '''


    '''

    r = ode(lotvolt_competition).set_integrator('dopri5')
    r.set_initial_value(pop, 0).set_f_params(coefs)

    #set output
    n1 = []
    n2 = []
    while r.successful() and r.t < timelength:
        r.integrate(r.t + d_t)
        n1.append(r.y[0])
        n2.append(r.y[1])

    return [n1, n2]



#Uncomment print statements to test the algebraic solution function
#print(get_lotvolt_comp_solution(1, 1, 1, 1)) #returns divide by 0 error (good)
#print(get_lotvolt_comp_solution(2, 1, 4, 4)) #return [1, 0] (good)
#print(get_lotvolt_comp_solution(4,4,2,5)) #return [-0, 1] (good enough)
#print(get_lotvolt_comp_solution(3, 4, 5, 6)) #good

init_pops = [0.3, 0.6] #[species1, species2]
coefs = [1, 2, 1, 3] #[a, b, c, d]
d_t = 1 #years
timelength = 100 #years
euler_pops, timeline = euler_solve_compmodel(init_pops, coefs, d_t, timelength)
rk45_pops = rk45_solve_compmodel(init_pops, coefs, d_t, timelength)
steady_state_solution = get_lotvolt_comp_solution(coefs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 6), sharex=True, sharey=True)

ax1.set_facecolor('ivory')
ax2.set_facecolor('ivory')

ax1.plot(euler_pops[0], color = 'limegreen',  label = "N1")
ax1.plot(euler_pops[1], color = 'sandybrown', ls = '--', label = "N2")
ax2.plot(rk45_pops[0], color = 'limegreen',  label = "N1")
ax2.plot(rk45_pops[1], color = 'sandybrown', ls = '--', label = "N2")
fig.suptitle("Lokta-Volterra Competition Model Integrated Solutions")
ax1.set_xlabel("Time (years)")
ax1.set_ylabel("Population Density")

ax1.set_title("Euler")
ax2.set_title("RK45")
ax1.legend()
ax2.legend()
plt.show()
