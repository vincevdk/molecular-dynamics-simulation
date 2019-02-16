import numpy as np
import matplotlib.pyplot as plt

from anim import make_3d_animation

# Physical values are defined globally
dim = 3
L = 10 # length of box
m = 1 # mass of particles


def build_matrices(n_timesteps, n_particles):
    vel = np.zeros(shape=(Nt, N_particle, dim), dtype=float)
    pos = np.zeros(shape=(Nt, N_particle, dim), dtype=float)
    F = np.zeros(shape=(Nt, N_particle, dim), dtype=float)
    dis = np.zeros(Nt)
    return(vel,pos,F,dis)


def initial_state(N_particles, vel,pos):
    #vel[0] = np.random.rand(N_particles, dim) * L
    pos[0] = np.random.rand(N_particles, dim) * L
    return(vel,pos)


def calculate_time_evolution(Nt, N_particle, vel, pos, dis, F):

    # distance = squareroot((dx)^2 + (dy)^2 + (dz)^2)
    dis[0] = (sum((pos[0,0,:] - pos[0,1,:])**2))**0.5 

    
    direction = np.zeros((N_particle, dim))
    direction[0,:] = -(pos[0,0,:] - pos[0,1,:])
    # minus sign for attractive forces                                       
    direction[1,:] = -(pos[0,1,:] - pos[0,0,:])

    F[0,0] = 3 * direction[0,:]
    F[0,1] = 3 * direction[1,:]
    print(F)

    for v in range(1,Nt):        
        pos[v] = (pos[v-1]+(1/Nt)*vel[v-1]) % L

        # velocity = vel[t-1] + 1/m * h * F[t-1]  p
        vel[v,:,:] = vel[v-1,:,:]+(1/m)*(1/Nt)*F[v-1,:,:]

        # direction of the line drawn between particles along which the force 
        # acts. For each particle a seperate direction vector is calculated.
        # the direction vector is not saved for different timesteps. 
        direction[0,:] = -(pos[v-1,0,:] - pos[v-1,1,:]) 
        # minus sign for attractive forces
        direction[1,:] = -(pos[v-1,1,:] - pos[v-1,0,:])

        # force along the line between particles
        F[v,0] = 3 * direction[0,:]
        F[v,1] = 3 * direction[1,:]

        # distance = squareroot((dx)^2 + (dy)^2 + (dz)^2) 
        dis[v] = (sum((pos[v,0,:] - pos[v,1,:])**2))**0.5

    return(vel,pos,dis)


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

    vel,pos,F,dis = build_matrices(Nt, N_particle)
    vel,pos = initial_state(N_particle,vel,pos)
    vel,pos,dis = calculate_time_evolution(Nt, N_particle, vel, pos, dis,F)    
    kin_energy = calculate_kinetic_energy(Nt,vel)
    
    anim = make_3d_animation(L, pos, delay=30, rotate_on_play=0)

    plt.figure()
    # plots the x-cordinate of particle 0 
    plt.plot(time,pos[:,0,0])
    plt.title('x-coordinate of particle 0')

    # plots the distance between the particles
    plt.figure()
    plt.plot(time,dis)
    plt.title('distance between the particles')

    plt.figure()
    plt.plot(time,kin_energy)
    plt.title('kinetic energy of all particles')
    plt.show()


