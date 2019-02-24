import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from scipy import spatial
from anim import make_3d_animation
from settings import *



def build_matrices(Nt, N_particle):
    vel = np.zeros(shape=(Nt, N_particle, dim), dtype=float)
    pos = np.zeros(shape=(Nt, N_particle, dim), dtype=float)
    potential_energy = np.zeros(shape=(Nt, N_particle), dtype = float)
    return(vel,pos,potential_energy)


def scaling_to_no_dim(time,vel,energy,L,epsilon,sigma,m):
    # initalize the unitless variables, which are defined at the start of the code
    time=time/(m*sigma**2/epsilon)**.5
    L=L/sigma
    vel=vel*np.sqrt(epsilon/m) 
    energy=energy*m/epsilon
    return(vel, energy,L,time)

def scaling_back(time,vel,energy,L,epsilon,sigma,m):
    force= force*sigma*m/epsilon # this one has to be done as to obtain the correct force in units at the end of the programm
    time=time*((m*sigma**2/epsilon)**.5)
    L=L*sigma
    vel=vel/(np.sqrt(epsilon/m)) 
    energy=energy/(m/epsilon)
    
    return(vel, energy,L,time,force)

def initial_state(N_particles, vel, pos, potential_energy):    
    energy =  -np.log(np.random.rand(N_particles,dim))*kb*temperature
    #inverting the probability function  to energy

    posneg = np.random.randint(2, size=(N_particles,dim))*2-1 
    #random number generator 1 or -1

    vel[0] = (2*energy/m)**.5*posneg #obtaining the velocity from the energy 
    pos[0] = np.random.rand(N_particles, dim) * L # generating the positions  

    min_dis, min_dir = calculate_minimal_distance_and_direction(N_particle,
                                                                pos[0])
    force = calculate_force(min_dir, min_dis)
    return(vel, pos, force)


def calculate_minimal_distance_and_direction(N_particles, pos_at_t):
    x = np.tile(pos_at_t,(N_particles,1,1))
    y = np.repeat(pos_at_t, N_particles, axis = 0)
    z = np.reshape(y,(N_particles,N_particles,dim))

    min_dir = np.array((x-z + L/2) % L - L/2)

    min_dis = np.array((np.sqrt(np.sum((min_dir**2), axis = 2))))
    min_dis = np.reshape(min_dis,(N_particles,N_particles,1))
    min_dis = np.repeat(min_dis,3,axis=2)
    return(min_dis, min_dir)


def calculate_potential_energy(N_particle,  pos_at_t, min_dis, min_dir):
    # dimensionless potential energy given in lecture notes
    potential_energy_at_t = 4*((min_dis)**12  - (min_dis)**6)
    return(potential_energy_at_t)


def calculate_force_matrix(min_dir_at_t, min_dis_at_t):
    # created a masked array to deal with division by zero
    F = ma.array(min_dir_at_t*(48*ma.power(min_dis_at_t,-14))-24*ma.power(min_dis_at_t,-8))
    return(F)
    
def calculate_force(min_dir_at_t, min_dis_at_t):
    force_matrix = calculate_force_matrix(min_dir_at_t, min_dis_at_t)
    total_force = (np.sum((force_matrix),axis = 1))
    return(total_force)

def calculate_time_evolution(Nt, N_particle, vel, pos, force):
    
    for v in range(1,Nt):        
        pos[v] = (pos[v-1]+(1/Nt)*vel[v-1]) % L
        
        vel[v,:,:] = vel[v-1,:,:]+(1/m)*(1/Nt) * force
        (min_dis, min_dir) = calculate_minimal_distance_and_direction(N_particle, pos[v])

        min_dis, min_dir = calculate_minimal_distance_and_direction(N_particle,
                                                                pos[v])
#        pot_energy[v] = calculate_potential_energy(N_particle,
#                                                         pos[v],
#                                                         min_dis,
#                                                         min_dir)
        force = calculate_force(min_dir, min_dis)
    return(vel,pos)


def calculate_kinetic_energy(n_timesteps, vel):
    # for each particle the kinetic energy is:
    # E_{kin} = 0.5 m (v_x^2 + v_y^2 + v_z^2)
    # the total kinetic energy is the sum of all particles
    kinetic_energy = 0.5 * m * (np.sum(vel[:,0,:]**2, axis=1) + np.sum(vel[:,1,:]**2, axis=1))
    print(kinetic_energy)
    return(kinetic_energy)




