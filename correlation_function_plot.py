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
from simulation import energy_plot

def pair_correlation_plot(bins, g_r):
    """ This function outputs a plot of the pair correlation function. 
    
    Parameters:
    --------
    bins:
    g_r:
    
    Result:
    -------
    
    """
    plt.figure()
    plt.title('pair correlation function')
    liquid, = plt.plot(bins[1:201], g_r[0], label="liquid" ) 
    solid, =plt.plot(bins[1:201], g_r[1], label="solid")
    gas, = plt.plot(bins[1:201], g_r[2], label="gas")
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.legend(handles=[liquid,solid,gas],loc=1)
    plt.show()


def calculate_correlation_function(temp, dens): 
    
    """ Calculation of the pair correlation function. It gives the probability, 
    given a reference particle, to find another particle at a distance r.
    
    Parameters:
    --------
    temp: array 
    dens: array
    
    Results: 
    --------
    g_r:
    bins:
    """

    cfg.temperature = temp
    cfg.density = dens
    cfg.L = (cfg.N_particle/cfg.density)**(1/3)

    if temp >= 3.0:
        cfg.h = 0.0002
        cfg.equilibration_time = np.arange(0, 30, 0.0002)
        cfg.simulation_time = np.arange(0, 20, cfg.h)
    else: 
        cfg.h = 0.002
        cfg.equilibration_time = np.arange(0, 20, 0.002)
        cfg.simulation_time = np.arange(0, 20, 0.002)

    # initialization
    (vel, pos, pot_energy, kin_energy, vir, sep_hist, temp) = build_matrices()
    pos_zero=pos
    vel, pos = initial_state(vel, pos)
    
    # equilibration phase
    pos, vel = equilibrate(vel,pos)

    # production phase
    (pot_energy, kin_energy, 
     virial, sep_hist, bins, temp) = calculate_time_evolution(vel, pos, pot_energy,
                                                  kin_energy, vir, sep_hist,temp)
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
    energy_plot(kin_energy/N_particle, pot_energy/N_particle, total_energy/N_particle)
    return(g_r, bins)

temperature = np.array((1.0,0.5,3.0))
density = np.array((0.8,1.2,0.7))
g_r = np.zeros((3,200))

for i in range(len(temperature)):
    temperature[i]
    g_r[i], bins = calculate_correlation_function(temperature[i],density[i])

# creat output plots     
pair_correlation_plot(bins, g_r)




 
 
