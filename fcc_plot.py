
import numpy as np 
from simulation import fcc_lattice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__=="__main__":
    cfg.N_particle = 108
    pos = np.zeros(shape=(cfg.N_particle, 3), dtype=float)
    pos = fcc_lattice(pos)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c='r', marker='o')

    plt.show()
