import numpy as np
import numpy.ma as ma

from src.functions import *
from src.config import *

print(N_particle)

def temperature_calculator(vel):
    """ function which calculates the temperature of the system. The 
    temperature is derived from the equipartition theorem
    E_kinetic_energy = (N-1)*3/2*kb*T
    
    Parameters: 
    ---------
    vel: array of size (N_particle, 3)
       The velocity of N particles in 3 dimensions. The first index of the
       array corresponds to a particle.
       
    Results:
    --------
    temperature_current: float
        Current temperature of the system
    """
    temperature_current=np.sum(vel**2)*1/(3*(N_particle-1))
    return(temperature_current)

def redistribute_velocity(vel,pos):
    """Calculates scaling factor. After letting the system equilibrate for 
    a while the kinetic energy is forced to a value corresponding to a certain 
    temperature
    
    Parameters:
    ---------
    vel: array of size (N_particle, 3)
       The velocity of N particles in 3 dimensions. The first index of the
       array corresponds to a particle.
    pos: array of size (N_particle, 3)
       The positon of N particles in 3 dimensions. The first index of the
       array corresponds to a particle.
       
    Results: 
    --------
    vel: array of size (N_particle, 3)
       The velocity of N particles in 3 dimensions. The first index of the
       array corresponds to a particle.
    """
    kinetic_energy = calculate_kinetic_energy(0, vel)
    rescaling_factor = ((N_particle - 1)*3/2*temperature/kinetic_energy)**.5
    vel = vel*rescaling_factor    
    return(vel)

def equilibrate(vel, pos):
    """function which rescales the velocity according to the required
    temperature. This is done by running the time evolution and checking if
    the temperature is correct at the end afterwhich the scaling factor
    scales it to the required temperature untill it converges

    Parameters:
    -----------
    vel: array of size (N_particle, 3)
       The velocity of N particles in 3 dimensions. The first index of the
       array corresponds to a particle.
    pos: array of size (N_particle, 3)
       The positon of N particles in 3 dimensions. The first index of the
       array corresponds to a particle.

    Results:
    -------
    pos:array of size (N_particle, 3)
       The positon of N particles in 3 dimensions. The first index of the array
       corresponds to a particle.
    vel:array of size (N_particle, 3)
       The velocity of N particles in 3 dimensions. The first index of the
       array corresponds to a particle.
    temperature_evolution: array of size (100)
       testfunction which shows what the temperature was before rescaling
       every loop. (not implemented in other parts of the code, except for
       tests)
   """
    min_dis, min_dir = calculate_minimal_distance_and_direction(pos)
    force = calculate_force(min_dir, min_dis)
    
    for v in range(len(equilibration_time)):
        vel = vel + h * force / 2
        pos = (pos + h * vel) % L
        min_dis, min_dir = calculate_minimal_distance_and_direction(pos)
        force = calculate_force(min_dir, min_dis)
        vel = vel + h * force / 2
        if v%10 == 0:
           vel = redistribute_velocity(vel,pos)
    vel = redistribute_velocity(vel,pos)
    return(pos, vel)
