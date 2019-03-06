import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from scipy import spatial
from anim import make_3d_animation
from config import *


def build_matrices():
    """Create the matrices used throughout the calculations.

    Results:
    --------
    vel: array of size (N_particle, 3)
       The velocity of N particles in 3 dimensions. The first index of the 
       array corresponds to a particle.      
    pos: array of size (N_particle, 3)
       The positon of N particles in 3 dimensions. The first index of the array
       corresponds to a particle.   
    potential_energy: array of size Nt
       The potential energy of all particles at each timestep.
    kinetic_energy: array of size Nt                                          
       The kintetic_energy of all particles at each timestep.  
    """
    vel = np.zeros(shape=(N_particle, dim), dtype=float)
    pos = np.zeros(shape=(N_particle, dim), dtype=float)
    potential_energy = np.zeros(Nt, dtype = float)
    kinetic_energy = np.zeros(Nt, dtype = float)
    return(vel,pos,potential_energy, kinetic_energy)

def fcc_lattice(pos_at_0):
    """Position all particles on a fcc lattice. First the number of unit cell
    is calculated which is used to calculate the distance between particles on 
    simple cubic lattice. Then particles are placed on a simple cubic lattice.
    Then using a shift the particles on the center of the cubic faces are 
    added.

    Parameters:
    -----------
    pos_at_0: array of size (N_paticle, 3)
        The positon of N particles in 3 dimensions. The first index of the 
        array corresponds to a particle. 

    Results:
    --------
    pos_at_0: array of size (N_paticle, 3)             
        The positon of N particles in 3 dimensions. The first index of the
        array corresponds to a particle.    
    """

    number_of_boxes = N_particle/4 
    distance_between_particles = ((L**3)/number_of_boxes)**(1/3)

    # Simple cubic
    x = np.arange(distance_between_particles/2, L, distance_between_particles)
    pos_at_0[0:int(number_of_boxes)] = np.array(np.meshgrid(x, x, x)).T.reshape(-1,3)  

    # add molecules on centre of cube faces  
    y = np.arange(distance_between_particles, 10, distance_between_particles)
    pos_at_0[int(number_of_boxes):2*int(N_particle/4)] = np.array(np.meshgrid(x,y,y)).T.reshape(-1,3)
    pos_at_0[2*int(N_particle/4):3*int(N_particle/4)] = np.array(np.meshgrid(y,y,x)).T.reshape(-1,3)
    pos_at_0[3*int(N_particle/4):N_particle] = np.array(np.meshgrid(y,x,y)).T.reshape(-1,3) 
    return(pos_at_0)


def initial_state(vel, pos,  pot_energy_t0, kin_energy_t0):
    """ Puts the system in an initial state.
    The velocities of the particles is calculated using the canonical 
    distribution. The position of the particles is initialized on a fcc 
    lattice. 
    Parameters:
    -----------
    vel: array of size (N_particle, 3)  
      The velocity of N particles in 3 dimensions. The first index of the 
      array corresponds to a particle.                                         
    pos: array of size (N_particle, 3) 
       The positon of N particles in 3 dimensions. The first index of the array
       corresponds to a particle.            
    pot_energy_t0: float
       The total potential energy at t = 0.
    kin_energy_t0: float
       The total kintetic energy at t = 0.

    Results:
    --------
    vel: array of size (N_particle, 3)
      The velocity of N particles in 3 dimensions. The first index of the 
      array corresponds to a particle.                                         
    pos: array of size (N_particle, 3) 
      The positon of N particles in 3 dimensions. The first index of the array       corresponds to a particle.                                               
    force: array of size (N_particle, N_particle)
    
    pot_energy_t0: float                                                     
       The total potential energy at t = 0.          
    kin_energy_t0: float                                                       
       The total kintetic energy at t = 0.                                     
                                                
    """
    energy =  -np.log(np.random.rand(N_particle,dim))*kb*temperature
    #inverting the probability function  to energy
    energy = energy*m/epsilon
    posneg = np.random.randint(2, size=(N_particle,dim))*2-1 
    #random number generator 1 or -1
    vel = (2*energy/m)**.5*posneg #obtaining the velocity from the energy 
    pos = fcc_lattice(pos)
    min_dis, min_dir = calculate_minimal_distance_and_direction(N_particle,
                                                                pos)
    force = calculate_force(min_dir, min_dis)

    pot_energy_t0 = calculate_potential_energy(pos, min_dis,min_dir, pot_energy_t0)

    kin_energy_t0 = calculate_kinetic_energy(kin_energy_t0, vel)
    return(vel, pos, force, pot_energy_t0, kin_energy_t0)


def calculate_minimal_distance_and_direction(N_particles, pos_at_t):
    dimension_added = np.tile(pos_at_t,(N_particles,1,1))
    transposed_matrix = np.repeat(pos_at_t, N_particles, axis = 0)
    transposed_matrix = np.reshape(transposed_matrix, (N_particles,N_particles,dim))
    min_dir = np.array((dimension_added - transposed_matrix + L/2) % L - L/2)
    min_dis = np.array((np.sqrt(np.sum((min_dir**2), axis = 2))))
    return(min_dis, min_dir)


def calculate_potential_energy(pos_at_t, min_dis, min_dir, potential_energy_at_t):
    # dimensionless potential energy given in lecture notes
    potential_energy_at_t = np.sum(4*(ma.power(min_dis,-12)  - ma.power(min_dis,-6)))
    return(potential_energy_at_t)


def calculate_force_matrix(min_dir_at_t, min_dis_at_t):
    # created a masked array to deal with division by zero
    F = ma.array(min_dir_at_t*(-48*ma.power(min_dis_at_t,-14))+24*ma.power(min_dis_at_t,-8))
    return(F)

    
def calculate_force(min_dir_at_t, min_dis_at_t):
    min_dis_at_t = np.reshape(min_dis_at_t,(N_particle,N_particle,1))
    min_dis_at_t = np.repeat(min_dis_at_t,3,axis=2)

    force_matrix = calculate_force_matrix(min_dir_at_t, min_dis_at_t)
    total_force = (np.sum((force_matrix),axis = 1))
    return(total_force)


def calculate_time_evolution(vel, pos, force, potential_energy,kinetic_energy):
    for v in range(1,Nt):        
        vel =  (vel+(1/(Nt*2)) * force)        
        pos = (pos + (1/Nt)*vel) % L
        min_dis, min_dir = calculate_minimal_distance_and_direction(N_particle,                                                                pos)

        force = calculate_force(min_dir, min_dis)
        vel = vel + (1/(Nt*2)) * force
        potential_energy[v] = calculate_potential_energy(pos, min_dis,min_dir, potential_energy[v])
        kinetic_energy[v] = calculate_kinetic_energy(kinetic_energy[v], vel)
    return(potential_energy,kinetic_energy)


def calculate_kinetic_energy(kinetic_energy_at_t, vel):
    # for each particle the kinetic energy is:
    # E_{kin} = 0.5 m (v_x^2 + v_y^2 + v_z^2)
    # the total kinetic energy is the sum of all particles
    kinetic_energy_at_t = 0.5 * np.sum(vel**2) # in units sigma
    return(kinetic_energy_at_t)

def calculate_total_energy(kin_energy,pot_energy):
    total_energy=kin_energy+pot_energy
    return(total_energy)




