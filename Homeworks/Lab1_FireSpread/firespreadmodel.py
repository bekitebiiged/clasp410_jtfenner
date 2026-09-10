'''
The following code is used to execute Lab01 in for CLaSP410 at University of Michigan
'''
#Import required libraries
import matplotlib.pyplot as plt
import numpy as np



#Declare constants to use throughout the model testing
num_x, num_y = 3, 3 #The X,Y gridsize for the Model
num_times = 3      #Number of "Times" the Model will iterate over
P_SPREAD = 1.0      #Probability of on-fire cell to spread to nearby forested cells
P_INIT_BARE = 0.0        #Probability of cell to start as bare spread
P_INIT_FIRE = 0.0       #Probability of cell to start on fire

#Set the Forest Model Element Values (for readability)
BARE = 1
FORESTED = 2
ON_FIRE = 3


def initialize_model_array(init_array, times):
    '''
    Take initial coniditions and create buffer around (insert row before/after and col before/after provided array) and add all timesteps needed
    Return that buffered array

    --------------
        INPUTS
    --------------
        init_array:
            2D Array of Initial State of Model
        times:
            Number of times to run the simulation (length of final array will be times-1 as it is 0 indexed)


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


#Complete Task 1 for the Lab
#* 3x3 grid
#* 100% chance of spread
#* zero initial bare spots
#* and only the center cell on fire.
#* Demonstrate correct behavior from iteration 0 to 1 and iteration 1 to 2.
#* Repeat this test with a larger grid that is wider than it is tall (e.g., 3x5).

# Create an initial frame for the tests 
first_forest = np.ones((num_x,num_y), dtype=int) * 2 
print("initial conditions:\n" , first_forest)
#create a 3x3 'forest' with P_SPREAD = 1, 0 initial bare spots, and 1 cell on fire (in the center)
forest = initialize_model_array(first_forest, num_times)
#Set center on fire (remember that array dimensions are now '1' indexed
forest[0, 2, 2] = ON_FIRE
#Use the actual array shape to define number of timesteps
times = forest.shape[0]
#Loop through each model time step (from the first to one before the last)
for time_step in range(times - 1):
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
                        if np.random.rand() < P_SPREAD:
                            pred_forest[coord] = ON_FIRE
    #Set each initally burning spot to BARE
    for coord in burning_spots:
        pred_forest[coord] = BARE
    #Put the predicted timestep as
    forest[time_step+1,:,:] = pred_forest

#Print the timesteps of the array
print("Final forest is: ", forest[:,slice(1,num_x+1),slice(1,num_y+1)])
