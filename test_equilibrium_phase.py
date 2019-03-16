import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from src.functions import *
from src.config import *
from src.observables import *
from src.initial_state import *
from src.production_phase import *
from src.equilibration_phase import *

def redistribute_velocity(vel,pos):
    kinetic_energy = calculate_kinetic_energy(0, vel)
    rescaling_factor = (N_particle - 1)*3/2*temperature/kinetic_energy
    vel = vel*rescaling_factor
    return(vel)

if __name__ == "__main__":
    vel, pos, pot_energy, kin_energy, drift_velocity, vir = build_matrices()
    vel, pos = initial_state(vel, pos)
    
    
    #####rescaling thingy
    min_dis, min_dir = calculate_minimal_distance_and_direction(pos)
    force = calculate_force(min_dir, min_dis)
        
    for v in range(total_equilibration_time):
        vel = vel + h * force / 2
        pos = (pos + h * vel) % L
        min_dis, min_dir = calculate_minimal_distance_and_direction(pos)
        force = calculate_force(min_dir, min_dis)
        vel = vel + h * force / 2
        if v%20 == 0:
           vel = redistribute_velocity(vel,pos)
           
    for v in range(len(equilibration_time)):
        print(v)
        if v%20 == 0:
            print("nu")

