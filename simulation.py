import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from src.functions import *
from src.config import *
from src.observables import *
from src.initial_state import *
from src.production_phase import *
from src.equilibration_phase import *

def seperation_distance_plot(min_dis):
    plt.figure()
    plt.title('seperation distance histogram')
    plt.hist(min_dis,50)
    plt.xlabel('distance')
    plt.ylabel('number of pairs')


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
    (vel, pos, pot_energy, kin_energy, 
     drift_velocity, vir, sep_hist) = build_matrices()

    vel, pos = initial_state(vel, pos)
    
    # equilibration phase
    pos, vel = equilibrate(vel,pos)

    # production phase
    (pot_energy, kin_energy,
     drift_velocity, virial, sep_hist) = calculate_time_evolution(vel,
                                                        pos,
                                                        pot_energy,
                                                        kin_energy,
                                                        drift_velocity,
                                                        vir, sep_hist)
    # data processing phase
    (simulation_time,
     kin_energy,
     pot_energy) = scaling_to_correct_dimensions(simulation_time,
                                                 kin_energy,
                                                 pot_energy)

    total_energy=calculate_total_energy(kin_energy,pot_energy)

    p = calculate_pressure(virial)
    p=p/(density*temperature*119.8)

#    drift = drift_velocity(vel,Nt,dim,drift)

#    seperation_distance_plot(seperation_histogram)

    energy_plot(kin_energy, pot_energy, total_energy)

    plt.figure()
    
    plt.title('drift velocity gass')
    
    
    plt.figure()
    
    plt.plot(simulation_time,p)
    plt.xlabel('time (s)')
    plt.ylabel('pressure')
        
    plt.show()
       
