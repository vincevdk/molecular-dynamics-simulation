import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from functions import *
from anim import make_3d_animation
from config import *

if __name__ == "__main__":

    vel, pos, pot_energy, kin_energy = build_matrices()
    vel, pos, force, pot_energy[0], kin_energy[0] = initial_state(vel, pos, pot_energy[0], kin_energy[0])

    pot_energy, kin_energy = calculate_time_evolution(vel, pos, force, pot_energy, kin_energy)

#    drift = drift_velocity(vel,Nt,dim,drift)

    plt.figure()
    plt.plot(time,kin_energy)
    plt.title('kinetic energy of all particles')

    plt.figure()
    plt.plot(time,pot_energy)
    plt.title('potential energy of all particles')
 #   plt.figure()
 #   plt.plot(time,drift)
 #   plt.title('momentum entire system')
    
    
    
    plt.show()
    
