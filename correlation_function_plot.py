import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from src.functions import *
from src.data_processing import *
from src.initial_state import *
from src.production_phase import *
from src.equilibration_phase import *
import time
from src import config as cfg

def pair_correlation_plot(bins, g_r):
    plt.figure()
    plt.title('pair correlation function')
    plt.plot(bins[1:201], g_r)
    plt.xlabel('r')
    plt.ylabel('g(r)')

def calculate_correlation_function(temp, dens): 

    cfg.temperature = temp
    cfg.density = dens
    cfg.L = (cfg.N_particle/cfg.density)**(1/3)  

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
    # data processing phase
    total_energy=calculate_total_energy(kin_energy,pot_energy)
    
    average_kin_energy_particle = time_average(kin_energy/N_particle)
    average_pot_energy_particle = time_average(pot_energy/N_particle)
    average_total_energy_particle = time_average(total_energy/N_particle)


    error_pot_energy = bootstrap(pot_energy/N_particle,100,100)

    p = calculate_pressure(virial)
    average_p = time_average(p)
    error_p = bootstrap(p,100,100)

    g_r = calculate_pair_correlation_function(sep_hist, bins)
    print('the potential energy is {0},with error {1} the pressure is {2}, with error {3}'.format(average_pot_energy_particle ,error_pot_energy, average_p,error_p))
    pair_correlation_plot(bins, g_r)

    return(g_r)

temperature = np.array((1,0.5,3.0))
pressure = np.array((0.8,1.2,0.3))

for i in range(len(temperature)):
    temperature[i]
    calculate_correlation_function(temperature[i],pressure[i])

# creat output plots     

plt.show()

 # print results to terminal


 
 
