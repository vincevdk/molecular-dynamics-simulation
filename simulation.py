import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from functions import *
from anim import make_3d_animation
from config import *

if __name__ == "__main__":

    vel, pos, pot_energy, kin_energy = build_matrices()
    vel, pos, force, pot_energy[0], kin_energy[0] = initial_state(vel, pos, pot_energy[0], kin_energy[0])


    pos,vel,temperature_evolution,pot_energy[0], kin_energy[0]=redistributing_velocity(vel, pos, force,pot_energy[0], kin_energy[0])
    
    pot_energy, kin_energy = calculate_time_evolution(vel, pos, force, pot_energy, kin_energy)
    
    
    
    time,kin_energy,pot_energy=scaling_to_correct_dimensions(time,kin_energy,pot_energy)
    total_energy=calculate_total_energy(kin_energy,pot_energy)
    
#    drift = drift_velocity(vel,Nt,dim,drift)

    plt.figure()
    
    plt.title('kinetic energy of all particles')
    
    plt.plot(time,kin_energy,label='kinetic energy')
    plt.plot(time,total_energy,label='total energy')
    plt.plot(time,pot_energy,label='potential energy')
    
    plt.xlabel('time (s)')
    plt.ylabel('energy (joule)')
    plt.grid(b=None, which='major', axis='both')
    plt.legend(loc='best')
    
    plt.show
       