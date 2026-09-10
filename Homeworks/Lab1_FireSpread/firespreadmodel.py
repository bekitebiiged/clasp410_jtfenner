'''
The following code is used to execute Lab01 in for CLaSP410 at University of Michigan
'''
#Import required libraries
import matplotlib.pyplot as plt
import numpy as np


#Set the Forest Model Element CONSTANTS (for readability)
BARE = 1
FORESTED = 2
ON_FIRE = 3


def initialize_model_array(init_array, times):
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
            Returns a buffered 3D array with inital conditions met

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
    #create an array that has two more rows and columns than the initial array
    buffered_array = np.zeros((times,shape[0]+2,shape[1]+2), dtype=int)
    #set the first and last rows as 'BARE'
    buffered_array[:,0,:] = buffered_array[:,-1,:] = BARE
    #set the first and last columns as 'BARE'
    buffered_array[:,:,0] = buffered_array[:,:,-1] = BARE
    #set the interior values of the array as the values of the inital array
    buffered_array[0,slice(1,shape[0]+1),slice(1,shape[1]+1)] = init_array

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

def model_fire_spread(initial_conditions, num_times, p_spread):
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
    #Return the full forest  
    return forest
#================================
#           Test 1
#================================
#Declare constants to use throughout the model testing
P_SPREAD = 1.0      #Probability of on-fire cell to spread to nearby forested cells
P_INIT_BARE = 0.0        #Probability of cell to start as bare spread
P_INIT_FIRE = 0.0       #Probability of cell to start on fire

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
print("initial conditions:\n" , first_forest)
#
test1_3by3_forest = model_fire_spread(first_forest, 3, P_SPREAD)
print("Final forest is:\n", test1_3by3_forest[:,slice(1,temp_num_x+1),slice(1,temp_num_y+1)])

temp_num_x = 3
temp_num_y = 5
temp_num_times = 3
first_forest = np.ones((temp_num_x,temp_num_y), dtype=int) * FORESTED
first_forest[temp_num_x//2,temp_num_y//2] = ON_FIRE
print("initial conditions:\n" , first_forest)
test1_3by5_forest = model_fire_spread(first_forest, 3, P_SPREAD)
print("Final forest is:\n", test1_3by5_forest[:,slice(1,temp_num_x+1),slice(1,temp_num_y+1)])

#Show that initial conditions are met
test_forests = [test1_3by3_forest, test1_3by5_forest]

fig, axes_2d = plt.subplots(2,3, figsize=(6,10))
for (ax_row, forest) in zip(range(axes_2d.shape[0]),test_forests):
    for (ax,curr_forest) in zip(axes_2d[ax_row,:],forest):
        print(forest.shape)
        ax.imshow(curr_forest)
plt.show()