#Import modules and such
from scipy.integrate import ode, RK45



#Define methods to be used
def lotvolt_competition(t, N, a=3, b=2, c=4, d=3):
    '''
    Calculates Lotka-Volterra population change rates for two species
    with populations N at time t
    ===============
        Inputs
    ===============
        t:
            time (in years)
        N:
            population density vector [pop1, pop2] [0,1]
        a, c:
            Reproduction rates of pops 1, 2 respectively
        b, d:
            scales impact of species on each other (2 on 1, 1 on 2)
    ===============
        Returns
    ===============
        dNdT:
            change rates [pop1 rate, pop2 rate], density per year
    '''
    #create empty vector to hold calulated rates
    dNdT = np.zeros(2)

    #Calculate RHS of Lotka Volterra Model
    dNdT[0] = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dNdT[1] = c*N[1]*(1-N[1]) - d*N[0]*N[1]

    return dNdT

def lotvolt_predprey(t, N, a=3, b=2, c=4, d=3):
    '''
    Calculates Predator-Prey Lotka-Volterra population change rates for two species
    with populations N at time t
    ===============
        Inputs
    ===============
        t:
            Time (in years)
        N:
            Population density vector [prey, predator] [0,1]
        a:
            Reproduction rates of prey
        b:
            coefficient describing hunted-ness of prey by predator
        c:
            Death rate of predators
        d:
            Coefficient describing growth of predator due to prey availablity

    ===============
        Returns
    ===============
        dNdT:
            change rates [prey rate, pred rate], density per year
    '''
    #create empty vector to hold calulated rates
    dNdT = np.zeros(2)

    #Calculate RHS of Lotka Volterra Model
    dNdT[0] = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dNdT[1] = -c*N[1]*(1-N[1]) + d*N[0]*N[1]

    return dNdT

#

while(foundSolution = False):

