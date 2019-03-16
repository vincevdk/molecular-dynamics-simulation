import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from src.functions import *
from src.config import *
from src.observables import *
from src.initial_state import *
from src.production_phase import *
from src.equilibration_phase import *


rescaling_time=np.ones(shape=1)

def redistribute_velocity(vel,pos,rescaling_time):
    kinetic_energy = calculate_kinetic_energy(0, vel)
    rescaling_factor = (N_particle - 1)*3/2*temperature/kinetic_energy
    vel = vel*rescaling_factor
    rescaling_time=np.append(rescaling_time,rescaling_factor)
    
    return(vel,rescaling_time)

if __name__ == "__main__":
    vel, pos, pot_energy, kin_energy, drift_velocity, vir = build_matrices()
    vel, pos = initial_state(vel, pos)
    
    
    #####rescaling thingy
    min_dis, min_dir = calculate_minimal_distance_and_direction(pos)
    force = calculate_force(min_dir, min_dis)
    average_velocity=np.sum(vel**2)
    
        
    for v in range(len(equilibration_time)):
        vel = vel + h * force / 2
        pos = (pos + h * vel) % L
        min_dis, min_dir = calculate_minimal_distance_and_direction(pos)
        force = calculate_force(min_dir, min_dis)
        vel = vel + h * force / 2
        if v%20 == 0:
           vel, rescaling_time = redistribute_velocity(vel,pos,rescaling_time)
           average_velocity=np.append(average_velocity,np.sum(vel**2))
           
           
    plt.figure()
    
    plt.plot(rescaling_time)
    plt.xlabel('time (s)')
    plt.ylabel('rescaling factor')
    
    plt.figure()
    plt.plot(rescaling_time)
        
    plt.show()
#           testing if the for loop is correct
#    for v in range(len(equilibration_time)):
#        print(v)
#        if v%20 == 0:
#            print("nu")

