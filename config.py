import numpy as np

# Physical values are defined globally

# Properties of Argon atoms and physical constants
m = 6.6 * 10**-26
sigma = 3.405*10**-10
kb = 1.38*10**-23
epsilon = 119.8*kb

# Chosen settings for simulation
dim = 3
temperature = 230
N_particle = 10
L = 1000  # size of the box in units sigma

# create an array "time" containing times at which we are calculating
timestep = 10*10**-14
dimensionless_timestep = timestep/(m*sigma**2/epsilon)**.5
Nt = 200 # number of timesteps  
time = np.arange(0,Nt*dimensionless_timestep-0.04,dimensionless_timestep)

