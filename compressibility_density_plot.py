import numpy as np
from simulation import *
import matplotlib.pyplot as plt

def compressibility_plot(density, compressibility):
    plt.figure()
    plt.plot(density,compressibility[0])
    plt.plot(density,compressibility[1])
    plt.xlabel('$\rho$')
    plt.ylabel('$\Beta P$')
    plt.label((2.74, 1.35))
    plt.show()

if __name__ == "__main__":
    temp = [2.74,1.35]
    density = [0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8]
    compressibility = np.zeros((len(temp),len(density)))
    for j in range (len(temp)):
        for i in range(len(compressibility)):
            _, compressibility[j][i] = run_simulation(temp[j],density[i],False)

    beta_rho = compressibility * density
    compressibility_plot(density, compressibility)
