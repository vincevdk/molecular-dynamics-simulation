import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from src.functions import *
from src.config import *
from src.observables import *
from src.initial_state import *
from src.production_phase import *
from src.equilibration_phase import *
import time

start = time.time()

def energy_plot(kin_energy, pot_energy, total_energy):
    plt.figure()

    plt.title('kinetic energy of all particles')

    plt.plot(simulation_time,kin_energy,label='kinetic energy')
    plt.plot(simulation_time,total_energy,label='total energy')
    plt.plot(simulation_time,pot_energy,label='potential energy')

    plt.xlabel('time (s)')
    plt.ylabel('energy (joule)')
    plt.grid(b=None, which='major', axis='both')
    plt.legend(loc='best')


if __name__ == "__main__":

    # initialization
    (vel, pos, pot_energy, kin_energy, vir, sep_hist) = build_matrices()
    pos_zero=pos
    vel, pos = initial_state(vel, pos)
    
    # equilibration phase
    pos, vel = equilibrate(vel,pos)

    # production phase
    (pot_energy, kin_energy, 
     virial, sep_hist, bins) = calculate_time_evolution(vel, pos, pot_energy,
                                                  kin_energy, vir, sep_hist)

    total_energy=calculate_total_energy(kin_energy,pot_energy)
    p = calculate_pressure(virial)
    g_r = calculate_pair_correlation_function(sep_hist, bins)

    # creat output plots     
    energy_plot(kin_energy/N_particle, pot_energy/N_particle, total_energy/N_particle)
    
    plt.figure()

    plt.title('pair correlation function')
    print(bins)
    plt.plot(bins[1:201], g_r)
    
    plt.figure()
    
    plt.plot(simulation_time,p)
    plt.xlabel('time (s)')
    plt.ylabel('pressure')
        
    plt.show()
       
    end = time.time()
    print("the total elapsed time is:",end - start)
