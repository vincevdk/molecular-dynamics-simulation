import numpy as np
import numpy.ma as ma
from src.config import *


def calculate_pressure(vir):
    """" Calculates the pressure devided by kb*T*rho
    
    Parameters:
    -----------
    vir: array of size Nt
       the virial is defined as: <0.5 Sum_(i,j) (r_(i,j) dU/dr_(ij))>
       where the sum is over i,j, i.e all particle pairs. r_(ij) is the 
       distance between a pair of particles

    Results:
    --------

    p: array of size Nt
       pressure of the system at each timestep

    """
    
    p = 1 - kb*temperature/(3*N_particle)*0.5*vir

    return(p)

def calculate_pair_correlation_function(seperation_histogram,bins):
    """ Calculates the pair correlation function using a histogram of 
    particles n(r) of all pairs within a distance [r,r + delta r], with 
    bin size delta r
    
    Parameters:
    -----------
    seperation_histogram: array of size (Nt, bins-1)
       For each timestep this array contains the number of particle pairs with
       a distance in the range of the size of the bins.

    bins: array of size bins+1
       Contains the edges of the bins. The difference between two successive 
       array elements is the size of each bin

    Results:
    --------
    pair_correlation_function: array of size bins
       defined as: (2 V <n(r)>) / (N(N-1) 4 pi r**2 delta r)

    """

    average_sep_histogram = np.sum(seperation_histogram, axis=0)/len(simulation_time)
    pair_correlation_function = np.array(200)
    delta_r = bins[1]-bins[0]
    pair_correlation_function = (2 * L**3 / (N_particle * (N_particle-1)) \
                                 * average_sep_histogram / (4 * np.pi* \
                                 bins[:-1]**2 * delta_r))

    return(pair_correlation_function)


def time_average(calculated_variable):
    """
    Parameters:
    -----------
    calculated_variable: array of size Nt

    Results:
    --------
    
    """
    average_of_variable = np.sum(calculated_variable)/len(simulation_time)

    return(average_of_variable)


def bootstrap(N_data_points, n_iterations, set_size):
    """
    Parameters:                                                               
    -----------                                       
    
    Results:                                      
    --------  
    """
    random_set = select_random_set(N_data_points, n_iterations, set_size)
    average_random_set = np.sum(random_set, axis = 1)/set_size
    standard_deviation = calculate_standard_deviation(average_random_set,n_iterations)
    return(standard_deviation)


def select_random_set(data_points, n_iterations, set_size):
    """
    """
    random_data_points = np.random.choice(data_points, size = (n_iterations, set_size))
    return(random_data_points)

def calculate_standard_deviation(average_random_set, n_iterations):
    standard_deviation = (np.sum(average_random_set**2)/n_iterations - (np.sum(average_random_set)/n_iterations)**2)**0.5
    return(standard_deviation)
