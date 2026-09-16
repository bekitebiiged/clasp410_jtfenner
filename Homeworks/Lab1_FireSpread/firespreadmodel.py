'''
The following code is used to execute Lab01 in for CLaSP410 at University of Michigan
'''
#Import required libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

plt.style.use('Solarize_Light2')

#Set the Forest Model Element CONSTANTS (for readability)
BARE = 1
FORESTED = 2
ON_FIRE = 3

#Declare constants to use throughout the model testing
P_SPREAD = 1.0              #Probability of on-fire cell to spread to nearby forested cells

def initialize_model_array(init_array, times = 1):
    '''
    Take initial coniditions and create buffer (insert row before/after and col before/after provided array)
    and add all timesteps defined.
    Return that buffered array

    --------------
        INPUTS
    --------------
        init_array:
            2D Array of Initial State of Model
        times:
            Number of times to run the simulation (length of final array will be times-1 as it is 0 indexed)

    --------------
        RETURNS
    --------------
        buffered_array:
            Returns a buffered 2D or 3D array with inital conditions met

    '''
    #Initialize a num_x by num_y by num_times forest grid with int dtype
    #Real forest is surrounded by 'Bare' Values
    # Example of '3x3' Forested Forest
    #
    #   [ 1, 1, 1, 1, 1 ]
    #   [ 1, 2, 2, 2, 1 ]
    #   [ 1, 2, 2, 2, 1 ]
    #   [ 1, 2, 2, 2, 1 ]
    #   [ 1, 1, 1, 1, 1 ]
    #

    #get initial array shape to define final array
    shape = init_array.shape
    #if we define number of timesteps return 3D array
    if times > 1:
        #create an 3D-times-deep array that has two more rows and columns than the initial array
        buffered_array = np.zeros((times,shape[0]+2,shape[1]+2), dtype=int)
        #set the first and last rows as 'BARE'
        buffered_array[:,0,:] = buffered_array[:,-1,:] = BARE
        #set the first and last columns as 'BARE'
        buffered_array[:,:,0] = buffered_array[:,:,-1] = BARE
        #set the interior values of the array as the values of the inital array
        buffered_array[0,slice(1,shape[0]+1),slice(1,shape[1]+1)] = init_array
    else:
        #create an 2D array that has two more rows and columns than the initial array
        buffered_array = np.zeros((shape[0]+2,shape[1]+2), dtype = int)
        #set the first and last rows as 'BARE'
        buffered_array[0,:] = buffered_array[-1,:] = BARE
        #set the first and last columns as 'BARE'
        buffered_array[:,0] = buffered_array[:,-1] = BARE
        #set the interior values of the array as the values of the inital array
        buffered_array[slice(1,shape[0]+1),slice(1,shape[1]+1)] = init_array



    #return the buffered array
    return buffered_array

#Function to Get Orthogonal Neighbors to
def get_neighbors(row, col):
    '''
    Returns touple of coordinate values touching a central point (no diagonals).
    Expected input is an array with 'dummy/ghost nodes' along the first/last row/cols
    Order is above, right, below, left

    INPUTS
        row:
        Index referring to a row of a 2D Numpy Array

        col:
        Index referring to a column of a 2D Numpy Array

    RETURNS:
        neighbors:
        Touple of toubles containing (row,col) indicies of neighboring elements



                    INDICIES TO BE RETURNED
    [   XXXXXXXXXXXX    ,   (row+1,col) , XXXXXXXXXXXXXX]
    [   (row, col-1)    ,   (row, col)  , (row, col+1)  ]
    [   XXXXXXXXXXXX    ,   (row-1, col), XXXXXXXXXXXXXX]


                    RETURN ORDER OF NEIGHBORS
    [   XXXXXXXXXXXX    ,       1        , XXXXXXXXXXXXXX]
    [        4          ,   (row, col)   ,       2       ]
    [   XXXXXXXXXXXX    ,       3        , XXXXXXXXXXXXXX]

    '''
    neighbors = ( (row+1, col), (row, col+1), (row-1, col), (row, col-1) )
    return neighbors

def test_fire_spread(initial_conditions, num_times, p_spread):
    '''
    Function to model Forest Fire Spread tests based on initial conditions and spreading probability

    --------------
        INPUTS
    --------------
        initial_conditions:
            2D Numpy array of initial conditions for a FOREST
        p_spread:
            Probability of fire to spread to Vegetated Regions
    --------------
        RETURNS
    --------------
        forest:
            3D Numpy array of calculated forest spread based on inputs
    '''

    #Create a 3D array based on the intitial conditions buffered by 'ghost nodes'
    forest = initialize_model_array(initial_conditions, num_times)

    #get dimensions of the initial array x,y (removing those ghost nodes)
    init_shape = initial_conditions.shape
    num_x = init_shape[0]
    num_y = init_shape[1]

    #Loop through each model time step (from the first to one before the last)
    for time_step in range(num_times - 1):
        #get the current model output as a 2D array (fixed time)
        curr_forest = forest[time_step, :, :]
        #create a predicted model output starting from the curr_forest
        pred_forest = np.copy(curr_forest)
        #create a list of spots that were initially burning
        burning_spots = []
        #loop through each element in the Real Forest (not the bounding BARE values)
        for curr_row in range(1,num_x+1):
            for curr_col in range(1,num_y+1):

                #Check to see if the current element is burning
                if(curr_forest[curr_row,curr_col] == ON_FIRE):
                    #Add the current burning spot to the inital burning list
                    burning_spots.append((curr_row, curr_col))
                    #find the neighboring grid spots
                    neighbors = get_neighbors(curr_row, curr_col)
                    #go through each neighboor to evaluate predicted value for next time step
                    for coord in neighbors:
                        #Fire spots can spread to Forested neighboors
                        if curr_forest[coord] == FORESTED:
                            if np.random.rand() < p_spread:
                                pred_forest[coord] = ON_FIRE
        #Set each initally burning spot to BARE
        for coord in burning_spots:
            pred_forest[coord] = BARE
        #Put the predicted timestep as
        forest[time_step+1,:,:] = pred_forest
    #Remove the bufferzone
    forest = forest[:,slice(1,num_x+1),slice(1,num_y+1)]
    return forest
#==============================================================
#        .-') _     ('-.    .-')    .-') _
#       (  OO) )  _(  OO)  ( OO ). (  OO) )
#       /     '._(,------.(_)---\_)/     '._        .---.
#       |'--...__)|  .---'/    _ | |'--...__)      /_   |
#       '--.  .--'|  |    \  :` `. '--.  .--'       |   |
#          |  |  (|  '--.  '..`''.)   |  |          |   |
#          |  |   |  .--' .-._)   \   |  |          |   |
#          |  |   |  `---.\       /   |  |          |   |
#          `--'   `------' `-----'    `--'          `---'
#===============================================================

# Create an initial frame for the tests
print("\t TESTING 3x3 MATRIX \n"
      +"\t FIRE IN MIDDLE\n")
#get the modeled output for Test 1
#create 3x3 matix with initial values of FORESTED
temp_num_x = 3
temp_num_y = 3
temp_num_times = 3
first_forest = np.ones((temp_num_x,temp_num_y), dtype=int) * FORESTED
first_forest[temp_num_x//2,temp_num_y//2] = ON_FIRE
#print("initial conditions:\n" , first_forest)
#
test1_3by3_forest = test_fire_spread(first_forest, 3, P_SPREAD)
#print("Final forest is:\n", test1_3by3_forest[:,slice(1,temp_num_x+1),slice(1,temp_num_y+1)])

temp_num_x = 4
temp_num_y = 10
temp_num_times = 3
first_forest = np.ones((temp_num_x,temp_num_y), dtype=int) * FORESTED
first_forest[temp_num_x//2,temp_num_y//2] = ON_FIRE
#print("initial conditions:\n" , first_forest)
test1_3by5_forest = test_fire_spread(first_forest, 3, P_SPREAD)
#print("Final forest is:\n", test1_3by5_forest[:,slice(1,temp_num_x+1),slice(1,temp_num_y+1)])

#Show that initial conditions are met
forest_cmap = ListedColormap(['tan', 'darkgreen', 'firebrick'])
test_forests = [test1_3by3_forest, test1_3by5_forest]
#https://stackoverflow.com/questions/27426668/row-titles-for-matplotlib-subplot
#Found subplot labeling help via code above
#View the two modeled output graphically
#

fig = plt.figure(constrained_layout=True)
fig.suptitle("Wildfire Model Validation")
subfigs = fig.subfigures(nrows=2, ncols = 1)
test_titles = [ '3 BY 3 GRID', '3 BY 5 GRID']
fig.legend()
for row, subfig in enumerate(subfigs):
    subfig.suptitle(test_titles[row])
    axs = subfig.subplots(nrows=1, ncols=3)
    for col, ax in enumerate(axs):
        ax.imshow(test_forests[row][col], cmap = forest_cmap)
        ax.grid(False)
        ax.set_title(f"T = {col}")
plt.show()


#===============================================================
#        .-') _     ('-.    .-')    .-') _
#       (  OO) )  _(  OO)  ( OO ). (  OO) )
#       /     '._(,------.(_)---\_)/     '._        .-----.
#       |'--...__)|  .---'/    _ | |'--...__)      / ,-.   \
#       '--.  .--'|  |    \  :` `. '--.  .--'      '-'  |  |
#          |  |  (|  '--.  '..`''.)   |  |            .'  /
#          |  |   |  .--' .-._)   \   |  |          .'  /__
#          |  |   |  `---.\       /   |  |         |       |
#          `--'   `------' `-----'    `--'         `-------'
##===============================================================

#function to generate an initial condition
def create_initial_conditions(num_x, num_y, init_onfire_prob=0.2, init_bare_prob=0):
    '''
    Function to create the initial conditions for the spread models.
    Creates an array of elements that are FORESTED/HEALTHY, BARE/IMMUNE, or ON_FIRE/SICk
    --------------
        INPUTS
    --------------
        num_x:
            size of initial forest in the x dimension
        num_y:
            size of initial forest in the y dimension
        init_onfire_prob:
            probability a square will initially be ON_FIRE
        init_bare_prob:
            probability a square will initially be BARE
    --------------
        RETURNS
    --------------
        initial_forest:
            num_x by num_y 2D Numpy array of initial forest conditions
    '''
    initial_conds = np.empty((num_x,num_y), dtype = int)
    #the following link was used as reference for 'total probability'
    #https://stackoverflow.com/questions/39582504/assigning-probabilities-to-items-in-python

    for row in range(num_x):
        for col in range(num_y):
            p = np.random.rand()
            #see if generated prob is less than the prob to start bare
            if p < init_bare_prob:
                initial_conds[row,col] = BARE
            #see if the generated prob is less than the ignite_prob (but greater than the bare prob)
            elif p < (init_bare_prob + init_onfire_prob):
                initial_conds[row,col] = ON_FIRE
            #make the element forested if not passed check for bare or on_fire
            else:
                initial_conds[row,col] = FORESTED

    return initial_conds


def model_fire_spread(initial_conditions, p_spread):
    '''
    Function to model Forest Fire Spread based on initial conditions and spreading probability

    --------------
        INPUTS
    --------------
        initial_conditions:
            2D Numpy array of initial conditions for a FOREST
        p_spread:
            Probability of fire to spread to Vegetated Regions
    --------------
        RETURNS
    --------------
        forest:
            3D Numpy array of calculated forest spread based on inputs
        time_to_burn:
            Number of time steps required for fire to stop (depth of 3D array)
        num_forested_remaining:
            The count of forested squares on last modeled forest
    '''

    #Create a 2D array based on the intitial conditions buffered by 'ghost nodes'
    curr_forest = initialize_model_array(initial_conditions)
    forest = [curr_forest]
    #get dimensions of the initial array x,y (removing those ghost nodes)
    init_shape = initial_conditions.shape
    #get the num_x and y for the og array (to use as reference)
    num_x = init_shape[0]
    num_y = init_shape[1]

    while(ON_FIRE in curr_forest):
        #create a predicted model output starting from the curr_forest
        pred_forest = np.copy(curr_forest)
        #create a list of spots that were initially burning
        burning_spots = []
        #loop through each element in the Real Forest (not the bounding BARE values)
        for curr_row in range(1,num_x+1):
            for curr_col in range(1,num_y+1):

                #Check to see if the current element is burning
                if(curr_forest[curr_row,curr_col] == ON_FIRE):
                    #Add the current burning spot to the inital burning list
                    burning_spots.append((curr_row, curr_col))
                    #find the neighboring grid spots
                    neighbors = get_neighbors(curr_row, curr_col)
                    #go through each neighboor to evaluate predicted value for next time step
                    for coord in neighbors:
                        #Fire spots can spread to Forested neighboors
                        if curr_forest[coord] == FORESTED:
                            if np.random.rand() < p_spread:
                                pred_forest[coord] = ON_FIRE
        #Set each initally burning spot to BARE
        for coord in burning_spots:
            pred_forest[coord] = BARE
        #Put the predicted timestep as
        forest.append(pred_forest)
        curr_forest = pred_forest

    #change the list of nparrays into an nparray
    forest = np.array(forest)
    #remove the ghost/buffer nodes
    forest = forest[:,slice(1,num_x+1),slice(1,num_y+1)]
    #Remove the bufferzone
    time_to_burn = forest.shape[0]
    #returns the count of forested elements in the final forest
    # works by summing up the "True [1]" values where the test is if element = 2
    num_forested_remaining = np.where(forest[-1,:,:] == FORESTED, True, False).sum()
    #forest = forest[:,slice(1,num_x+1),slice(1,num_y+1)]
    return forest, time_to_burn, num_forested_remaining

#Test the new model compared to the original:
#test = np.ones((4,4), dtype = int) * 2
#test[3,3] = 3
#test[1,1] = 1
#test_output = model_fire_spread(test, 0.2)
#print(test_output)


#Test the percentages
#x = create_initial_forest(100,100,ignite_prob=0.2, bare_prob=0.5)
# works by summing up the "True [1]" values where the test is if element = 2
#percent = ( np.where(x == ON_FIRE, True, False).sum() )
#print(percent)

# Vary p_spread from 0 to 1, try ten trials per p_spread value.
# Store the p_spread and final times (plot as p_spread vs final time chart)
# Store the num_forested and final times (plot as num_forested vs final time)

#Declare constants to use throughout the model testing
P_SPREAD = 1.0              #Probability of on-fire cell to spread to nearby forested cells
P_INIT_BARE = 0.0           #Probability of cell to start as bare spread
P_INIT_FIRE = 0.3           #Probability of cell to start on fire
num_trials_per_spread = 10  #Number of trials for step (chose ten because increments of p_spread are .1
                            #Need at least 10 trials for fire to spread once reliably

#Create list of probabilities [0,1]
fire_spread_probs = np.arange(0, 1, 0.1)
varying_spread_results = {
    "Burning Time": [],
    "Remaining Forest Squares": [],
    "Spread Probability": []
}

varying_bare_results = {
    "Burning Time": [],
    "Remaining Forest Squares": [],
    "Initial Bare Probability": []
}

#All trials will have 100 by 100 forest size
num_x = 10
num_y = 10
for spread_prob in np.arange(0,1.1,0.1):
    for trial_idx in np.arange(num_trials_per_spread):
        init_cond = create_initial_conditions(num_x, num_y, init_onfire_prob=P_INIT_FIRE)
        curr_model = model_fire_spread(init_cond, spread_prob)
        #store the time_to_burn and num_forested remaining
        varying_spread_results['Burning Time'].append(curr_model[1]-1)
        varying_spread_results['Remaining Forest Squares'].append(curr_model[2])
        varying_spread_results['Spread Probability'].append(spread_prob)

for bare_prob in np.arange(0,1.1,0.1):
    for trial_idx in np.arange(num_trials_per_spread):
        init_cond = create_initial_conditions(num_x=num_x, num_y=num_y,
                                          init_onfire_prob=P_INIT_FIRE, init_bare_prob=bare_prob)
        curr_model = model_fire_spread(init_cond, P_SPREAD)
        #store the time_to_burn and num_forested remaining
        varying_bare_results['Burning Time'].append(curr_model[1]-1)
        varying_bare_results['Remaining Forest Squares'].append(curr_model[2])
        varying_bare_results['Initial Bare Probability'].append(bare_prob)

fig = plt.figure(figsize=(10,8), constrained_layout=True)
fig.suptitle("Fire Spread Probability and Bare Forest Spot influence on Wildfire Spread")
subfigs = fig.subfigures(nrows=2, ncols = 1)
titles = [ 'Varying Fire Spread Probability [FULL FOREST START]', 'Varying Initial Forest Density [P_SPREAD = 100]']
variables = [varying_spread_results, varying_bare_results]
x_vars = ['Spread Probability', 'Initial Bare Probability']
y_vars = ['Burning Time', 'Remaining Forest Squares']
fig.legend()
for row, subfig in enumerate(subfigs):
    subfig.suptitle(titles[row])
    axs = subfig.subplots(nrows=1, ncols=2)
    for col, ax in enumerate(axs):
        ax.scatter(variables[row][x_vars[row]], variables[row][y_vars[col]])
        ax.set_title(f"{y_vars[col]}")
        ax.set_xlabel(x_vars[row])
        ax.set_ylabel(y_vars[col])
plt.show()

#===============================================================
#        .-') _     ('-.    .-')    .-') _
#       (  OO) )  _(  OO)  ( OO ). (  OO) )
#       /     '._(,------.(_)---\_)/     '._        .-----.
#       |'--...__)|  .---'/    _ | |'--...__)      /  -.   \
#       '--.  .--'|  |    \  :` `. '--.  .--'      '-' _'  |
#          |  |  (|  '--.  '..`''.)   |  |            |_  <
#          |  |   |  .--' .-._)   \   |  |         .-.  |  |
#          |  |   |  `---.\       /   |  |         \ `-'   /
#          `--'   `------' `-----'    `--'          `----''
#===============================================================

#We will now modify the code to allow instead look at illness spread instead of wildfire spread

HEALTHY = FORESTED  #both equal to 2
SICK = ON_FIRE      #both equal to 3
IMMUNE = BARE       #both equal to 1
DEAD = 0            #new constant

#new check to add is 'p_fatal' or 'p_survive' which is the likelihood a SICK person will DIE or IMMUNE


def model_illness_spread(initial_conditions, times = 0, p_spread = 1, p_fatal = 0):
    '''
    Function to model Zombie Virus Spread based on initial conditions and spreading probability

    --------------
        INPUTS
    --------------
        initial_conditions:
            2D Numpy array of initial conditions for a FOREST
        p_spread:
            Probability of illness to spread to healthy cells
        p_fatal:
            Probability sick cell will perish (1 - p_survive)
    --------------
        RETURNS
    --------------
        illness_spread:
            3D Numpy array of calculated illness spread
        time_to_spread:
            Number of time steps required for illness to spread (depth of 3D array)
        num_healthy:
            The count of healthy cells at end of model
        num_alive:
            The count of immune cells at end of model
        num_dead:
            The count of dead cells at the end of the model
    '''

    #Create a 2D array based on the intitial conditions buffered by 'ghost nodes'
    curr_model_slice = initialize_model_array(initial_conditions)
    illness_spread = [curr_model_slice]
    #get dimensions of the initial array x,y (removing those ghost nodes)
    init_shape = initial_conditions.shape
    #get the num_x and y for the og array (to use as reference)
    num_x = init_shape[0]
    num_y = init_shape[1]

    if times == 0:
        while(SICK in curr_model):
            #create a predicted model output starting from the curr_forest
            pred_model_slice = np.copy(curr_model_slice)
            #create a list of spots that were initially sick
            initially_sick = []
            #loop through each element in the curr_model_slice (not the bounding IMMUNE values)
            for curr_row in range(1,num_x+1):
                for curr_col in range(1,num_y+1):

                    #Check to see if the current element is SICK
                    if(curr_model_slice[curr_row,curr_col] == SICK):
                        #Add the current burning spot to the inital burning list
                        initially_sick.append((curr_row, curr_col))
                        #find the neighboring grid spots
                        neighbors = get_neighbors(curr_row, curr_col)
                        #go through each neighboor to evaluate predicted value for next time step
                        for coord in neighbors:
                            #Fire spots can spread to Forested neighboors
                            if curr_model_slice[coord] == HEALTHY:
                                if np.random.rand() < p_spread:
                                    pred_model_slice[coord] = SICK
            #Check if each initially SICK cell becomes IMMUNE or DEAD
            for coord in initially_sick:
                p = np.random.rand()
                if p < p_fatal:
                    pred_model_slice[coord] = DEAD
                else:
                    pred_model_slice[coord] = IMMUNE
            #Put the predicted timestep as
            illness_spread.append(pred_model_slice)
            curr_model_slice = pred_model_slice
    else:
        for time in range(times):
            #create a predicted model output starting from the curr_forest
            pred_model_slice = np.copy(curr_model_slice)
            #create a list of spots that were initially sick
            initially_sick = []
            #loop through each element in the curr_model_slice (not the bounding IMMUNE values)
            for curr_row in range(1,num_x+1):
                for curr_col in range(1,num_y+1):

                    #Check to see if the current element is SICK
                    if(curr_model_slice[curr_row,curr_col] == SICK):
                        #Add the current burning spot to the inital burning list
                        initially_sick.append((curr_row, curr_col))
                        #find the neighboring grid spots
                        neighbors = get_neighbors(curr_row, curr_col)
                        #go through each neighboor to evaluate predicted value for next time step
                        for coord in neighbors:
                            #Fire spots can spread to Forested neighboors
                            if curr_model_slice[coord] == HEALTHY:
                                if np.random.rand() < p_spread:
                                    pred_model_slice[coord] = SICK
            #Check if each initially SICK cell becomes IMMUNE or DEAD
            for coord in initially_sick:
                p = np.random.rand()
                if p < p_fatal:
                    pred_model_slice[coord] = DEAD
                else:
                    pred_model_slice[coord] = IMMUNE
            #Put the predicted timestep as
            illness_spread.append(pred_model_slice)
            curr_model_slice = pred_model_slice

    #change the list of nparrays into an nparray
    illness_spread = np.array(illness_spread)
    #remove the ghost/buffer nodes
    illness_spread = illness_spread[:,slice(1,num_x+1),slice(1,num_y+1)]
    #Get the runtime (1 minus depth as first time is T = 0)
    time_to_spread = illness_spread.shape[0] - 1

    #Get remaining count of HEALTHY and IMMUNE cells
    #works by summing up the "True" (equal 1) values where the test is if element = 2
    num_healthy = np.where(illness_spread[-1,:,:] == HEALTHY, True, False).sum()
    num_immune = np.where(illness_spread[-1:,:] == IMMUNE, True, False).sum()
    num_dead = np.where(illness_spread[-1,:,:] == DEAD, True, False).sum()
    #forest = forest[:,slice(1,num_x+1),slice(1,num_y+1)]
    return illness_spread, time_to_spread, num_healthy, num_immune, num_dead

#Test the illness spread using similar tests to the forest one

# Create an initial frame for the tests
print("\t TESTING 6x6 MATRIX \n"
      +"\t SICK IN MIDDLE\n")
#get the modeled output for Test 1
#create 3x3 matix with initial values of FORESTED
temp_num_x = 3
temp_num_y = 3
temp_num_times = 5
init_test_illness = np.ones((temp_num_x,temp_num_y), dtype=int) * HEALTHY
init_test_illness[temp_num_x//2,temp_num_y//2] = SICK
#print("initial conditions:\n" , first_forest)
#Test with full illness spread and 50% fatality rate
test_3by3_illness = model_illness_spread(init_test_illness, times = temp_num_times, p_spread=1, p_fatal=0.5)
#print("Final forest is:\n", test1_3by3_forest[:,slice(1,temp_num_x+1),slice(1,temp_num_y+1)])

temp_num_x = 20
temp_num_y = 30
temp_num_times = 5
init_test_illness = create_initial_conditions(temp_num_x,temp_num_y, init_bare_prob= 0.3, init_onfire_prob=0.2)
#print("initial conditions:\n" , first_forest)
test_20by20_illness = model_illness_spread(init_test_illness, times = temp_num_times, p_spread = 1, p_fatal = 0.3)
#print("Final forest is:\n", test1_3by5_forest[:,slice(1,temp_num_x+1),slice(1,temp_num_y+1)])

#Create color map with [0 = white, 1 = tan, 2 = green, 3 = firebrick]
illness_cmap = ListedColormap(['white','tan', 'darkgreen', 'firebrick'])
test_illness_models = [test_3by3_illness, test_20by20_illness]
#https://stackoverflow.com/questions/27426668/row-titles-for-matplotlib-subplot
#Found subplot labeling help via code above
#View the two modeled output graphically

fig = plt.figure(constrained_layout=True)
fig.suptitle("Illness Model Validation")
subfigs = fig.subfigures(nrows=2, ncols = 1)
test_titles = [ f'6 BY 6 GRID', '20 BY 30 GRID']
fig.legend()
for row, subfig in enumerate(subfigs):
    subfig.suptitle(test_titles[row])
    axs = subfig.subplots(nrows=1, ncols=temp_num_times)
    for col, ax in enumerate(axs):
        curr_model = test_illness_models[row][0]
        print(row, col)
        ax.imshow(curr_model[col,:,:],vmin = 0, vmax = 3, cmap = illness_cmap)
        ax.grid(False)
        ax.set_title(f"T = {col}")
plt.show()


