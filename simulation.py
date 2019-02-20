import numpy as np
import matplotlib.pyplot as plt

from anim import make_3d_animation

# Physical values are defined globally
dim = 3
L = 10 # length of box
m = 1 # mass of particles
kb = 10
temperature = 2
N_particle = 2

def build_matrices(n_timesteps, n_particles):
    vel = np.zeros(shape=(Nt, N_particle, dim), dtype=float)
    pos = np.zeros(shape=(Nt, N_particle, dim), dtype=float)
    potential_energy = np.zeros(shape=(Nt, N_particle), dtype = float)
    return(vel,pos,potential_energy)


def initial_state(N_particles, vel, pos, potential_energy):    
    energy =  -np.log(np.random.rand(N_particles,dim))*kb*temperature
    #inverting the probability function  to energy

    posneg = np.random.randint(2, size=(N_particles,dim))*2-1 
    #random number generator 1 or -1

    vel[0] = (2*energy/m)**.5*posneg #obtaining the velocity from the energy 
    pos[0] = np.random.rand(N_particles, dim) * L # generating the positions  

    min_dis, min_dir = calculate_minimal_distance_and_direction(N_particle,
                                                                pos[0])
    potential_energy[0] = calculate_potential_energy(N_particles, 
                                                     pos[0],
                                                     min_dis, 
                                                     min_dir)

    force = calculate_force(min_dir, min_dis)
    return(vel,pos, potential_energy, force)


def calculate_minimal_distance_and_direction(N_particle, pos_at_t):
        
    min_dir = np.array(((pos_at_t[0,:] - pos_at_t[1,:] + L/2) % L - L/2,
                        (pos_at_t[1,:] - pos_at_t[0,:] + L/2) % L - L/2))

    min_dis = np.array((np.sqrt(np.sum((min_dir[0]**2), axis = 0)), 
                        np.sqrt(np.sum((min_dir[1]**2), axis = 0))))

    return(min_dis, min_dir)


def calculate_potential_energy(N_particle,  pos_at_t, min_dis, min_dir):

    # dimensionless potential energy
    potential_energy_at_t = 4*((min_dis)**12  - (min_dis)**6)

    return(potential_energy_at_t)

def calculate_force(min_dir_at_t, min_dis_at_t):
    # dimensionless force (equation 8.22 in chapter 8 molecular dynamics)
    F = np.array((min_dir_at_t[0]*(48*(min_dis_at_t[0])**-14)-24*(min_dis_at_t[0])**-8, min_dir_at_t[1]*(48*(min_dis_at_t[1])**-14)-24*(min_dis_at_t[1])**-8))
    return(F)
    
def calculate_time_evolution(Nt, N_particle, vel, pos, pot_energy, force):
    
    for v in range(1,Nt):        
        pos[v] = (pos[v-1]+(1/Nt)*vel[v-1]) % L
        
        # velocity = vel[t-1] + 1/m * h * F[t-1]  p
        vel[v,:,:] = vel[v-1,:,:]+(1/m)*(1/Nt) * force
        (min_dis, min_dir) = calculate_minimal_distance_and_direction(N_particle, pos[v])

        min_dis, min_dir = calculate_minimal_distance_and_direction(N_particle,
                                                                pos[v])
        pot_energy[v] = calculate_potential_energy(N_particle,
                                                         pos[v],
                                                         min_dis,
                                                         min_dir)
        force = calculate_force(min_dir, min_dis)
    return(vel,pos, pot_energy)


def calculate_kinetic_energy(n_timesteps, vel):
    # for each particle the kinetic energy is:
    # E_{kin} = 0.5 m (v_x^2 + v_y^2 + v_z^2)
    # the total kinetic energy is the sum of all particles
    kinetic_energy = 0.5 * m * (np.sum(vel[:,0,:]**2, axis=1) + np.sum(vel[:,1,:]**2, axis=1))
    print(kinetic_energy)
    return(kinetic_energy)


if __name__ == "__main__":    
    N_particle = 2
    Nt = 2000 # number of timesteps  
    time = np.linspace(1,Nt,Nt)

    vel,pos, pot_energy = build_matrices(Nt, N_particle)
    vel, pos, pot_energy, force = initial_state(N_particle, vel, pos, pot_energy)
    vel,pos, pot_energy = calculate_time_evolution(Nt, N_particle, vel, pos, pot_energy,force)    
    kin_energy = calculate_kinetic_energy(Nt,vel)
    
    anim = make_3d_animation(L, pos, delay=30, rotate_on_play=0)

    plt.figure()
    # plots the x-cordinate of particle 0 
    plt.plot(time,pos[:,0,0])
    plt.title('x-coordinate of particle 0')

    plt.figure()
    plt.plot(time,kin_energy)
    plt.title('kinetic energy of all particles')
    plt.show()


