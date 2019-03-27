import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from src.functions import *
from src import config
from src.config import *
from src.data_processing import *
from src.initial_state import *
from src.production_phase import *
from src.equilibration_phase import *
import time
from src import config as cfg

start = time.time()

def energy_plot(kin_energy, pot_energy, total_energy):
    """ Plots kinetic energy, potential energy and total energy in a 
    single plot

    Parameters:
    -----------
    kin_energy: array of size Nt (number of simulation timesteps)
        kinetic energy at each timestep
    pot_energy: array of size Nt 
        potential energy at each timestep
    total_energy: array of size Nt 
        total energy at each timestep
    """
    plt.figure()

    plt.title('energy per particle')

    plt.plot(simulation_time,kin_energy,label='kinetic energy')
    plt.plot(simulation_time,total_energy,label='total energy')
    plt.plot(simulation_time,pot_energy,label='potential energy')

    plt.xlabel('time (s)')
    plt.ylabel('energy ($\epsilon$)')
    plt.grid(b=None, which='major', axis='both')
    plt.legend(loc='best')


def pair_correlation_plot(bins, g_r):
    """ Creates a plot with on the y-axis the pair correlation function 
    and on the x-axis the distance between particles

    parameters:
    ----------
    bins: array of size 201
        Contains the edges of the bins. The difference between two successive  
        array elements is the size of each bin. Note that it can be a different
        size then 201 (201 is the case in this program).
    g_r: array of size 200
        The pair correlation function
    """
    plt.figure()
    plt.title('pair correlation function')
    plt.plot(bins[0:-1], g_r)
    plt.xlabel('r')
    plt.ylabel('g(r)')


def compressibility_plot(p):
    """
    Parameters:
    -----------
    p: array of Nt
    """
    plt.figure()
    plt.plot(simulation_time,p)
    plt.xlabel('time (s)')
    plt.ylabel('pressure')


def run_simulation(temp, dens, N_par = 32, plots = True):
    """ Runs the simulation in the following steps:
    initialization -> equilibration -> production phase -> data processing ->
    output results

    Parameters:
    -----------
    temp: float
    dens: float
    N_par : int
        optional input
    plots: True or False
        optional input
    """
    cfg.temperature = temp
    cfg.density = dens
    cfg.N_particle = N_par
    cfg.L = (N_particle/density)**(1/3)  # size of the box in units sigma    

    # initialization
    (vel, pos, pot_energy, kin_energy, 
     vir, sep_hist, t_current) = build_matrices()

    vel, pos = initial_state(vel, pos)
    
    # equilibration phase
    pos, vel = equilibrate(vel,pos)

    # production phase
    (pot_energy, kin_energy, virial, 
     sep_hist, bins, temp) = calculate_time_evolution(vel, pos, pot_energy,
                                                      kin_energy, vir, 
                                                      sep_hist, t_current)

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

    # output results
    if plots == True:
        energy_plot(kin_energy/N_particle, pot_energy/N_particle, 
                    total_energy/N_particle)
        pair_correlation_plot(bins, g_r)
        compressibility_plot(p)
    
    print('''the potential energy is {0}, with error {1} the compressibility
          is {2}, with error {3}'''.format(average_pot_energy_particle,
                                   error_pot_energy, average_p,error_p))

    end = time.time()
    print("the total elapsed time is:",end - start)
    return(g_r, average_p)


if __name__ == "__main__":
    # input is given here
    temp = 1.0
    dens = 0.8
    N_particle = 32
    
    run_simulation(temp, dens, N_particle)
    plt.show()
