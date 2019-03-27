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
    bins: array of size 201
       Contains the edges of the bins. The difference between two successive  
       array elements is the size of each bin
    g_r: array of size (3,200)
       the pair correlation function defined as: 
       (2 V ⟨n(r)⟩) / (N(N-1) 4 pi r**2 delta r) 
       for three different input combinations of pressure and temperature
     
    Result:
    -------
    Outputs the correlation function for three input combinations of pressure 
    and temperature
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
    """ Calculation of the pair correlation function. It gives the 
    probability, given a reference particle, to find another particle at a 
    distance r. This function is similair to 
    
    Parameters:
    --------
    temp: array 
        input temperature for
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
    (pot_energy, kin_energy, virial, 
     sep_hist, bins, temp) = calculate_time_evolution(vel, pos, pot_energy,
                                                      kin_energy, vir, 
                                                      sep_hist,temp)
    # data processing phase
    total_energy=calculate_total_energy(kin_energy,pot_energy)
    
    average_kin_energy_particle = time_average(kin_energy/N_particle)
    average_pot_energy_particle = time_average(pot_energy/N_particle)
    average_total_energy_particle = time_average(total_energy/N_particle)

    error_pot_energy = bootstrap(pot_energy/N_particle,100)

    p = calculate_compressibility(virial)
    average_p = time_average(p)
    error_p = bootstrap(p,100)

    g_r = calculate_pair_correlation_function(sep_hist, bins)
    return(g_r, bins)

temperature = np.array((1.0,0.5,3.0))
density = np.array((0.8,1.2,0.7))
g_r = np.zeros((3,200))

for i in range(len(temperature)):
    temperature[i]
    g_r[i], bins = calculate_correlation_function(temperature[i],density[i])

# creat output plots     
pair_correlation_plot(bins, g_r)




 
 
