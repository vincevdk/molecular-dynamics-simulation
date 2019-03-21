import numpy as np
import numpy.ma as ma
from src.config import *

def calculate_pressure(vir):
    """" Calculates the pressure devided by kb*T*rho
    
    parameters:
    ------
    vir: array of size simulationtime
    
    result:
    ------
    p: float
        pressure of the system for each timestep
    """
    
    p = 1 - kb*temperature/(3*N_particle)*0.5*vir

    return(p)

def calculate_pair_correlation_function(seperation_histogram,bins):
    """
    """
    average_sep_histogram = np.sum(seperation_histogram, axis=0)/len(simulation_time)
    print(average_sep_histogram)
    print(np.sum(average_sep_histogram), 'average_sep')
    pair_correlation_function = np.array(200)
    delta_r = bins[1]-bins[0]
    pair_correlation_function = 2 * L**3 / (N_particle * (N_particle-1)) * average_sep_histogram / (4 * np.pi* bins[:-1]**2 * delta_r)
    return(pair_correlation_function)

def time_average(calculated_variable):
    """
    """
    average_of_variable = calculated_variable/len(simulation_time)

    return(average_of_variable)
