import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from functions import *
from anim import make_3d_animation
from config import *

if __name__ == "__main__":

    vel, pos, pot_energy = build_matrices(Nt, N_particle)
    vel, pos, force = initial_state(N_particle, vel, pos, pot_energy)

    vel, pos = calculate_time_evolution(Nt, N_particle, vel, pos, force)
    kin_energy = calculate_kinetic_energy(Nt,vel)
#    drift = drift_velocity(vel,Nt,dim,drift)

    anim = make_3d_animation(L, pos, delay=30, rotate_on_play=0)

    plt.figure()

    # plots the x-cordinate of particle 0
    plt.plot(time,pos[:,0,0])
    plt.title('x-coordinate of particle 0')

    plt.figure()
    plt.plot(time,kin_energy)
    plt.title('kinetic energy of all particles')
    
 #   plt.figure()
 #   plt.plot(time,drift)
 #   plt.title('momentum entire system')
    
    
    
    plt.show()
    
