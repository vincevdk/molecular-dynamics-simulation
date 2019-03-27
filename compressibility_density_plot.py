import numpy as np
from simulation import *
import matplotlib.pyplot as plt

def compressibility_plot(density, compressibility):
    """ Plots the compressibility for different densities
    
    Parameters: 
    --------
    density: array
    
    Result:
    --------
    plot of the density versus the compressibility
    """
    plt.figure()
    plt.plot(density,compressibility[0])
    plt.plot(density,compressibility[1])
    plt.xlabel(r'$\rho $')
    plt.ylabel(r'$\beta P$')
    plt.legend(('$T = 2.74$', '$T = 1.35$'))
    plt.show()

if __name__ == "__main__":
    temp = [2.74,1.35]
    density = [0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8]
    beta_P = np.zeros((len(temp),len(density)))

    compressibility = np.zeros((len(temp),len(density)))
    for j in range (len(temp)):
        for i in range(len(density)):
            _, compressibility[j][i] = run_simulation(temp[j],density[i],False)
            print(compressibility[j][i],'compressibility')
            print(compressibility[0] * density)
    beta_P[0] = compressibility[0] * density
    beta_P[1] = compressibility[1] * density
    compressibility_plot(density, beta_P)
