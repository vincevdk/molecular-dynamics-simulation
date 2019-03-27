import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from src import config as cfg

def build_matrices():
    """Create the matrices used throughout the calculations.

    Results:
    --------
    vel: array of size (N_particle, 3)
       The velocity of N particles in 3 dimensions. The first index of the
       array corresponds to a particle.
    pos: array of size (N_particle, 3)
       The positon of N particles in 3 dimensions. The first index of the
       array corresponds to a particle.
    potential_energy: array of size Nt
       The potential energy of all particles at each timestep.
    kinetic_energy: array of size Nt
       The kintetic energy of all particles at each timestep.
    vir: array of size Nt
       The sum (over all pairs) of the distance times the force between two 
       pairs.
    sep_hist: array of size (Nt, 200)
       Histogram with 200 bins of the seperation distance between particles
    """

    vel = np.zeros(shape=(cfg.N_particle, cfg.dim), dtype=float)
    pos = np.zeros(shape=(cfg.N_particle, cfg.dim), dtype=float)
    potential_energy = np.zeros(cfg.Nt, dtype=float)
    kinetic_energy = np.zeros(cfg.Nt, dtype=float)
    vir = np.zeros(cfg.Nt, dtype=float)
    t_current = np.zeros(cfg.Nt, dtype=float)
    sep_hist = np.zeros(shape=(cfg.Nt,200), dtype=float)

    return(vel, pos, potential_energy, kinetic_energy, 
           vir, sep_hist, t_current)


def fcc_lattice(pos_at_0):
    """Position all particles on a fcc lattice. First the number of unit cell
    is calculated which is used to calculate the distance between particles on
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
    number_of_boxes = cfg.N_particle / 4
    distance_between_particles = ((cfg.L**3) / number_of_boxes)**(1 / 3)

    # Simple cubic
    x = np.arange(distance_between_particles / 2,
                  cfg.L, distance_between_particles)
    pos_at_0[0:int(number_of_boxes)] = np.array(
        np.meshgrid(x, x, x)).T.reshape(-1, 3)

    # add molecules on center of cube faces
    y = np.arange(
        distance_between_particles,
        cfg.L + distance_between_particles / 2,
        distance_between_particles)
    pos_at_0[int(number_of_boxes):2 * int(cfg.N_particle / 4)] 
             = np.array(np.meshgrid(x, y, y)).T.reshape(-1, 3)
    pos_at_0[2 * int(cfg.N_particle / 4):3 * int(cfg.N_particle / 4)] 
             = np.array(np.meshgrid(y, y, x)).T.reshape(-1, 3)
    pos_at_0[3 * int(cfg.N_particle / 4):cfg.N_particle] 
             = np.array(np.meshgrid(y, x, y)).T.reshape(-1, 3)

    return(pos_at_0)


def initial_state(vel, pos):
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
       The positon of N particles in 3 dimensions. The first index of the
       array corresponds to a particle.

    Results:
    --------
    vel: array of size (N_particle, 3)
      The velocity of N particles in 3 dimensions. The first index of the
      array corresponds to a particle.
    pos: array of size (N_particle, 3)
      The positon of N particles in 3 dimensions. The first index of the
      array corresponds to a particle.
    """
    
    energy = -np.log(np.random.rand(cfg.N_particle, cfg.dim)) * cfg.kb * cfg.temperature*119.8
    # inverting the probability function  to energy
    energy = energy * cfg.m / cfg.epsilon  # dimensionless
    # random number generator 1 or -1 
    pos_or_neg = np.random.randint(2, size=(cfg.N_particle, cfg.dim)) * 2 - 1

    # obtaining the velocity from the energy
    vel = (2 * energy / cfg.m)**.5 * pos_or_neg

    pos = fcc_lattice(pos)

    return(vel, pos)
