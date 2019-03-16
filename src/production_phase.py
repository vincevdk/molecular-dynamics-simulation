import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from scipy import spatial
from src.config import *
from src.observables import *
from src.functions import *

def calculate_time_evolution(vel, pos, potential_energy,
                             kinetic_energy, vir, seperation_histogram):
    """Uses velocity verlet

    Parameters:
    -----------
    vel: array of size (N_particle, 3)
       The velocity of N particles in 3 dimensions. The first index of the
       array corresponds to a particle.
    pos: array of size (N_particle, 3)
       The positon of N particles in 3 dimensions. The first index of the
       array corresponds to a particle.
    vir: array of size Nt
       The sum (over all pairs) of the distance times the force between two
       pairs.
    seperation_histogram: array of size (Nt, 200)
       Histogram with 200 bins of the seperation distance between particles 
     
    Results:
    -------
    potential_energy: array of size Nt                                         
       The potential energy of all particles at each timestep. 
    kinetic_energy: array of size Nt                                           
       The kintetic_energy of all particles at each timestep. 
    vir: array of size Nt
       The sum (over all pairs) of the distance times the force between two 
       pairs.
    seperation_histogram: array of size (Nt, 200)                              
       Histogram with 200 bins of the seperation distance between particles 
    """
    min_dis, min_dir = calculate_minimal_distance_and_direction(pos)           
    force = calculate_force(min_dir, min_dis)                                  
    potential_energy[0] = calculate_potential_energy(min_dis, 
                                                     potential_energy[0])

    kinetic_energy[0] = calculate_kinetic_energy(kinetic_energy[0], vel) 
    vir[0] = virial_theorem(pos)
    seperation_histogram[0], _ = np.histogram(min_dis, 200)

    for v in range(1, Nt):
        vel = vel + h * force / 2
        pos = (pos + h * vel) % L
        min_dis, min_dir = calculate_minimal_distance_and_direction(pos)
        force = calculate_force(min_dir, min_dis)
        vel = vel + h * force / 2
        potential_energy[v] = calculate_potential_energy(min_dis, 
                                                         potential_energy[v])

        kinetic_energy[v] = calculate_kinetic_energy(kinetic_energy[v], vel)
        vir[v] = virial_theorem(pos)

        seperation_histogram[v], _ = np.histogram(min_dis, 200)  

    return(potential_energy, kinetic_energy, vir, seperation_histogram )
