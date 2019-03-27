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
    plt.figure()
    plt.title('pair correlation function')
    plt.plot(bins[0:-1], g_r)
    plt.xlabel('r')
    plt.ylabel('g(r)')


def pressure_plot(p):
    plt.figure()
    plt.plot(simulation_time,p)
    plt.xlabel('time (s)')
    plt.ylabel('pressure')

def run_simulation(temp, dens, plots = True):

    cfg.temperature = temp
    cfg.density = dens
    cfg.L = (N_particle/density)**(1/3)  # size of the box in units sigma    

    # initialization
    (vel, pos, pot_energy, kin_energy, vir, sep_hist, t_current) = build_matrices()

    vel, pos = initial_state(vel, pos)
    
    # equilibration phase
    pos, vel = equilibrate(vel,pos)

    # production phase
    (pot_energy, kin_energy, 
     virial, sep_hist, bins, temp) = calculate_time_evolution(vel, pos, pot_energy,
                                                  kin_energy, vir, sep_hist, t_current)

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
    # creat output plots     
    if plots == True:
        energy_plot(kin_energy/N_particle, pot_energy/N_particle, total_energy/N_particle)
        pair_correlation_plot(bins, g_r)
        pressure_plot(p)
    
    # print results to terminal
    print('the potential energy is {0},with error {1} the pressure is {2}, with error {3}'.format(average_pot_energy_particle ,error_pot_energy, average_p,error_p))

    end = time.time()
    print("the total elapsed time is:",end - start)
    return(g_r, average_p)

if __name__ == "__main__":
    temp = 1.0
    dens = 0.8
    run_simulation(temp,dens)
    plt.show()
