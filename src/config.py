import numpy as np

# Chosen settings for simulation 
dim = 3
temperature = 100
N_particle = 32
L = 10  # size of the box in units sigma 
timestep = 10**-14

# Properties of Argon atoms and physical constants
m = 6.6 * 10**-26
sigma = 3.405*10**-10
kb = 1.38*10**-23
epsilon = 119.8*kb

# Dimensionless variables
temperature = temperature/119.8 # in units (epsilon/kb)
h = timestep/(m*sigma**2/epsilon)**.5 # dimensionless timestep
total_equilibration_time = 10 # in units of [(m * sigma^2 / epsilon)^0.5]
total_simulation_time = 20 # in units of [(m * sigma^2 / epsilon)^0.5]   
equilibration_time = np.arange(0, total_equilibration_time,h)
simulation_time = np.arange(total_equilibration_time, 
                            total_equilibration_time+total_simulation_time,h)

Nt = len(simulation_time)
