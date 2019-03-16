import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from scipy import spatial
from src.config import *
from src.observables import *
from src.functions import *

def calculate_time_evolution(vel, pos, potential_energy,
                             kinetic_energy, drift_velocity, 
                             vir, seperation_histogram):

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
        drift_velocity[v, :] = np.sum(vel, axis=0)
        vir[v] = virial_theorem(pos)

        seperation_histogram[v], _ = np.histogram(min_dis, 200)  

    return(potential_energy, kinetic_energy, 
           drift_velocity, vir, seperation_histogram )
