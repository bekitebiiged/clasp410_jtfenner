'''
This code was used for the lab Population models, chaos, ordinary differential equations
and so much more
follow this code to model the Lokta-Volterra equations of ecology and population...
as well as simple weather model to simulate deterministic chaos
'''
#importing necessary packages
import numpy as np
import matplotlib
matplotlib.use('TkAgg') #setting up back end
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import ode #used later for ODE45
from scipy.integrate import solve_ivp

'''
The twist!!
Orcas consider seals to be both competitors and prey. In this situation, we are separating the competitor and predator-prey relationships for easier conceptualization
in reality, a more realistic model would contain a combined predator-prey and competitor functions into one model.
'''


'''
#hand-coded competition model: Euler Method
🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭
'''

'''
orca_init = 0.3 #predator initial population density
seal_init = 0.6 #prey initial population density
N_init = [orca_init, seal_init]
dt = 1 #time step width in years
num_steps = 100

#here, i am defining my function arctic that is describing the functions of competition dynamics
def arctic(N):
    a = 1 #constant
    b = 2 #constant
    c = 1 #constant
    d = 3 #constant 
    N1,N2 =N[0],N[1] #we have made a vector of our species 1 and species 2 population densities, so it can return as one value as opposed to having to sparate it
    dN1dt = a*N1*(1-N1)-b*N1*N2 #changes of population density of species 1, orca, for competition
    dN2dt = c*N2*(1-N2)-d*N1*N2 #changes of population density of species 2, seal, for competition
    return np.array([dN1dt,dN2dt]) #report the changes of pop density after a time step with a width size defined by dt
dNdt=arctic(N_init) #the initial change in pop densities over first time step in our func artic

#prepare figure
plt.figure(figsize=(8,8))

#storing history of data outside of loop
N=np.copy(N_init)
time_history=[0.0] #time bin to graph later each step rather than it being erased
popden_history=[np.copy(N)]

#Euler method loop of Lokta-Voltera Competition Model
for t in range(num_steps):
    #N_init = np.copy(N_curr) #in our loop, we change our initial condition to be the next step, as we learned form the euler method
    dNdt=arctic(N)
    N_next = N + dt * dNdt

    # Prevent populations from going below 0. This was an important check of code soundness because we can't have negative population densities.
    N_next = np.maximum(0, N + dt * dNdt)
    #therefore, a species density at 0 stays at 0 and doesn't go beyond, aka it is extinct

    N = np.copy(N_next)

    #append new data into our history
    time_history.append((t+1)*dt)
    popden_history.append(np.copy(N))

#convert the history into an array for easy slicing
popden_history = np.array(popden_history)

#plotting complete trajectory for current dt
plt.plot(time_history, popden_history[:,0], label="Orca (N1)", color="blue")
plt.plot(time_history,popden_history[:,1], label="Seal (N2)", color="orange")
#Plot aesthetics
plt.xlabel('Time (Years)')
plt.ylabel('Normalized Population Density (1 = Carrying Capacity')
plt.title('Population Density: Competition of Orcas and Seals')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
'''

'''
#hand-coded predatory-prey model: Euler Method
🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭
'''
'''
orca_init = 0.3 #predator initial population density
seal_init = 0.6 #prey initial population density
N_init = [seal_init, orca_init]
dt = .01 #time in years
num_steps = 200

def arctic(N):
    a = 1 #constant
    b = 2 #constant
    c = 1 #constant
    d = 3 #constant 
    N1,N2 =N[0],N[1] #N1 is seal, N2 is orca
    dN1dt =a*N1-b*N1*N2 #prey
    dN2dt =-c*N2+d*N1*N2 #predator
    return np.array([dN1dt,dN2dt]) #report the changes of pop density after a time step with a width size defined by dt
dNdt=arctic(N_init) #the initial change in pop densities over first time step in our func artic
    
#prepare figure
plt.figure(figsize=(8,8))
    
#storing history of data outside of loop
N=np.copy(N_init)
time_history=[0.0] #time bin to graph later each step rather than it being erased
popden_history=[np.copy(N)]
    
#Euler method loop of Lokta-Voltera Competition Model
for t in range(num_steps):
    #N_init = np.copy(N_curr) #in our loop, we change our initial condition to be the next step, as we learned form the euler method
    dNdt=arctic(N)
    N_next = N + dt * dNdt
    
    # Prevent populations from going below 0. This was an important check of code soundness because we can't have negative population densities.
    N_next = np.maximum(0, N + dt * dNdt)
    #therefore, a species density at 0 stays at 0 and doesn't go beyond, aka it is extinct
    
    N = np.copy(N_next)
    
    #append new data into our history
    time_history.append((t+1)*dt)
    popden_history.append(np.copy(N))
    
#convert the history into an array for easy slicing
popden_history = np.array(popden_history)
    
#plotting complete trajectory for current dt
plt.plot(time_history, popden_history[:,0], label="Seal (N1)", color="orange")
plt.plot(time_history,popden_history[:,1], label="Orca (N2)", color="blue")
#Plot aesthetics
plt.xlabel('Time (Years)')
plt.ylabel('Normalized Population Density (1 = Carrying Capacity)')
plt.title('Population Density: Predator/Prey Relationship of Orcas and Seals')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


#plotting phase diagram of species 1 and species 2 
plt.plot(popden_history[:,0], popden_history[:,1])
plt.title('Eulers Method Phase Space Seal (N1) and Orca (N2) Population Density')
plt.xlabel('Normalized Seal Population Density (N1)')
plt.ylabel('Normalized Orca Population Density (N2)')
plt.show()
'''

'''
Runge Kutta 4 solver
about rk45: a scipy product that can solve for an initial value problem for a system of ODEs
'''

'''
#rk45 competition model
🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭
'''

'''
orca_init = 0.3 #species 1 initial population density
seal_init = 0.6 #species 2 initial population density
N_init = [orca_init, seal_init]
num_steps = 100
t_span = (0, num_steps) #time step width in years
t_eval = np.linspace(0, 50, 500)

def arctic(t, N): #now defining function of t and pop den, since we are not manually deciding dt
    a = 1 #constant
    b = 2 #constant
    c = 1 #constant
    d = 3 #constant 
    K = 1 #carrying capacity
    N1,N2 =N[0],N[1] #orca species 1, seal species 2
    dN1dt = a*N1*(1-N1)-b*N1*N2 #changes of population density of species 1, orca, for competition
    dN2dt = c*N2*(1-N2)-d*N1*N2 #changes of population density of species 2, seal, for competition
    return ([dN1dt,dN2dt]) #report the changes of pop density after a time step with a width size defined by dt


sol = solve_ivp(
    fun=arctic, #function
    t_span=t_span, #time span
    y0=N_init, #initial pop density vector
    method='RK45', #package using
    t_eval=t_eval #eval values at these time steps
)

plt.figure(figsize=(8, 5))
plt.plot(sol.t, sol.y[0], label='Orcas (N1)', color='blue')
plt.plot(sol.t, sol.y[1], label='Seals (N2)', color='orange')
plt.title('RK45 Arctic Competition Model Simulation')
plt.xlabel('Time (Years)')
plt.ylabel('Normalized Population Density (Carrying Capacity = 1)')
plt.legend()
plt.grid(True)
plt.show()

#plotting phase diagram of species 1 and species 2 
plt.plot(sol.y[0], sol.y[1])
plt.title('RK45 Method Phase Space Competition between Orcas (N1) and Seals (N2)')
plt.xlabel('Normalized Orca Population Density (N1)')
plt.ylabel('Normalized Seal Population Density (N2)')
plt.show()
'''



'''
#rk45 predator-prey model
🦭🦭🦭🦭🦭🦭🦭🦭🦭🦭
'''

'''
orca_init = 0.3 #predator initial population density
seal_init = 0.6 #prey initial population density
N_init = [seal_init, orca_init]
num_steps = 100
t_span = (0, num_steps) #time steps
t_eval = np.linspace(0, 50, 100)

def arctic(t, N):
    a = 1 #constant
    b = 2 #constant
    c = 1 #constant
    d = 3 #constant 
    N1,N2 =N[0],N[1] #N1 is seal, N2 is orca
    dN1dt =a*N1-b*N1*N2 #prey, seal
    dN2dt =-c*N2+d*N1*N2 #predator, orca
    return ([dN1dt,dN2dt]) #report the changes of pop density after a time step with a width size defined by dt

#here is where we enable using the scipy.integrate import solve_ivp. first. we tell the solution that we to solve_ivp
#then we give it the name of our function (mine is arctic), the t_span must be a range of time iterations (num_steps)
#y0 are the initial values of the populations
#the method we are using si RK45
#we can use t_eval to report pop densities at specific time steps
sol = solve_ivp(
    fun=arctic,
    t_span=t_span,
    y0=N_init,
    method='RK45',
    t_eval=t_eval
)

plt.figure(figsize=(8, 5))
plt.plot(sol.t, sol.y[0], label='Seals (N1)', color='blue')
plt.plot(sol.t, sol.y[1], label='Orcas (N2)', color='orange')
plt.title('RK45 Arctic Predator/Prey Model Simulation')
plt.xlabel('Time (Years)')
plt.ylabel('Normalized Population Density (Carrying Capacity = 1)')
plt.legend()
plt.grid(True)
plt.show()

#plotting phase diagram of species 1 and species 2 
plt.plot(sol.y[0], sol.y[1])
plt.xlabel('Normalized Seal Population Density (N1)')
plt.ylabel('Normalized Orca Population Density (N2)')
plt.show()
'''


