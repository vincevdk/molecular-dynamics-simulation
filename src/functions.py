import numpy as np
import numpy.ma as ma

from src import config as cfg


def calculate_minimal_distance_and_direction(pos_at_t):
    """ Calculates the difference in x,y,z coordinates between each pair of
    particles and the distance between the particles

    Parameters
    -----------
    pos_at_t: array of size (N_particle, 3)
       The positon of N particles in 3 dimensions. The first index of the
       array corresponds to a particle.

    Results:
    --------
    min_dir: array of size (N_particle, N_particle, 3)
       A single matrix  entry [i,j,l] is the difference in coordinate l
       (x,y or z) between i and j (i_l - j_l). Thus the matrix has zeros on
       the diagonal as the distance between the particle and itself is zero.
    min_dis: array of size (N_particle, N_particle)
       A single matrix entry [i,j] is the distance between particle i and j.
       An entry is calculated using the difference in coordinates from the
       min_dis matrix with the formula: sqrt(dx^2+dy^2+dz^2).
    """
    threshold = 10**(-4)
    dimension_added = np.tile(pos_at_t, (len(pos_at_t), 1, 1))
    transposed_matrix = np.repeat(pos_at_t, len(pos_at_t), axis=0)
    transposed_matrix = np.reshape(
       transposed_matrix, (len(pos_at_t), len(pos_at_t), cfg.dim))

    min_dir = np.array(
        (dimension_added - transposed_matrix + cfg.L / 2) %
        cfg.L - cfg.L / 2)
    min_dis = np.array((np.sqrt(np.sum((min_dir**2), axis=2))))
    np.fill_diagonal(min_dis, 1)
    min_dis[min_dis < threshold] = threshold
    np.fill_diagonal(min_dis, 0)

    return(min_dis, min_dir)


def calculate_potential_energy(min_dis_at_t, potential_energy):
    """ Dimensionless Lennard-Jones potential energy
    
    Parameters:
    -----------
    min_dis_at_t: array of size (N_particle, N_particle)
       A single matrix entry [i,j] is the distance between particle i and j.
    potential_energy: float

    Results:
    --------
    potential_energy: float
       Total potential energy of the system
    """

    potential_energy = np.sum(4 * (ma.power(min_dis_at_t, -12) 
                                   - ma.power(min_dis_at_t, -6))) / 2
    return(potential_energy)


def calculate_force_matrix(min_dir_at_t, min_dis_at_t):
    """ Calculates the force for particles using the derivative of the Lennard
    Jones potential. A masked array is used to deal with division by zero.

    Parameters:
    -----------
    min_dis_at_t: array of size (N_particle, N_particle)
       A single matrix entry [i,j] is the distance between particle i and j.
    min_dir_at_t: array of size (N_particle, N_particle, 3)
       A single matrix  entry [i,j,l] is the difference in coordinate l
       (x,y or z) between i and j (i_l - j_l). Thus the matrix has zeros on
       the diagonal as the distance between the particle and itself is zero.

    Results:
    --------
    F: array of size (N_particle, N_particle, 3)
       A single matrix  entry [i,j,l] is the force component of coordinate l
       (x,y or z) between i and j (i_l - j_l). Thus the matrix has zeros on
       the diagonal as the distance between the particle and itself is zero.
    """
    F = ma.array(min_dir_at_t * (- 48 * ma.power(min_dis_at_t, -14) 
                                 + 24 * ma.power(min_dis_at_t, -8)))
    return(F)


def calculate_force(min_dir_at_t, min_dis_at_t):
    """ Calculates the force

    Parameters:
    -----------
    min_dis_at_t: array of size (N_particle, N_particle)
       A single matrix entry [i,j] is the distance between particle i and j.
    min_dir_at_t: array of size (N_particle, N_particle, 3)
       A single matrix  entry [i,j,l] is the difference in coordinate l
       (x,y or z) between i and j (i_l - j_l). Thus the matrix has zeros on
       the diagonal as the distance between the particle and itself is zero.

    Results:
    --------
    total_force: array of size (N_particle,3)
       A single matrix entry [i,j] is the force component of coordinate j on
       particle i .
    """
    min_dis_at_t = np.reshape(
        min_dis_at_t, (len(min_dir_at_t), len(min_dir_at_t), 1))
    min_dis_at_t = np.repeat(min_dis_at_t, 3, axis=2)

    force_matrix = calculate_force_matrix(min_dir_at_t, min_dis_at_t)
    total_force = (np.sum((force_matrix), axis=1))
    return(total_force)


def calculate_kinetic_energy(kinetic_energy, vel):
    """Calculates the total kinetic energy of the system
    for each particle: E_kin = 0.5 m (v_x^2 + v_y^2 + v_z^2)
    the total kinetic energy is the sum of all particles   
    
    Parameters:
    -----------
    kinetic_energy: float
       kinetic energy of the system
    vel: array of size (N_particle, 3)
       The velocity of N particles in 3 dimensions. The first index of the
       array corresponds to a particle.

    Results:
    --------
    kinetic_energy: float
       kinetic energy of the system
    """

    kinetic_energy_at_t = 0.5 * np.sum(vel[:, 0]**2 
                                       + vel[:, 1]**2 
                                       + vel[:, 2]**2)
    return(kinetic_energy_at_t)


def calculate_total_energy(kin_energy, pot_energy):
    """" Calculates the total energy of the system
    total energy = kinetic energy + potential energy
    Parameters:
    -----------
    kinetic energy: float
        kinetic energy of the system
    potential energy: float
        potential energy of the system
    Results:
    ---------
    total_energy: float
    """
    total_energy = kin_energy + pot_energy
    return(total_energy)


def virial_theorem(pos):
    """
    Parameters:
    -----------
    pos: array of size (N_particle, 3) 
       The positon of N particles in 3 dimensions. The first index of the      
       array corresponds to a particle. 
    Results:
    --------
    vir: array of size len(simulation_time)                                    
       the virial is defined as: <0.5 Sum_(i,j) (r_(i,j) dU/dr_(ij))>          
       where the sum is over i,j, i.e all particle pairs. r_(ij) is the        
       distance between a pair of particles 
    """
    min_dis, min_dir = calculate_minimal_distance_and_direction(pos)
    vir = ma.array(
        min_dis**2 * (-48 * ma.power(min_dis, -14) 
         + 24 * ma.power(min_dis, -8)))
    vir = np.sum(vir)
    return(vir)
