import numpy as np

# Chosen settings for simulation 
dim = 3
N_particle = 108
#density=N_particle/L**3
density=0.80 #units of 1/sigma**3
L = (N_particle/density)**(1/3)  # size of the box in units sigma 
timestep = 10**-14

# Properties of Argon atoms and physical constants
m = 6.6 * 10**-26
sigma = 3.405*10**-10
kb = 1.38*10**-23
epsilon = 119.8*kb

# Dimensionless variables
temperature = 1 # in units (epsilon/kb)
h = timestep/(m*sigma**2/epsilon)**.5 # dimensionless timestep
total_equilibration_time = 10 # in units of [(m * sigma^2 / epsilon)^0.5]
total_simulation_time = 20 # in units of [(m * sigma^2 / epsilon)^0.5]   
equilibration_time = np.arange(0, total_equilibration_time,h)
simulation_time = np.arange(total_equilibration_time, 
                            total_equilibration_time+total_simulation_time,h)

Nt = len(simulation_time)
