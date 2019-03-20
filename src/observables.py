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



#def calculate_pair_correlation_function():
