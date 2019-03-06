import numpy as np

# Properties of Argon atoms and physical constants
m = 6.6 * 10**-26
sigma = 3.405*10**-10
kb = 1.38*10**-23
epsilon = 119.8*kb

# Chosen settings for simulation
dim = 3
temperature = 1000
N_particle = 108
L = 10  # size of the box in units sigma

# create an array "time" containing times at which we are calculating
timestep = 10**-14
h = 0.001
#h = timestep/(m*sigma**2/epsilon)**.5 # dimensionless_timestep

Nt = 500 # number of timesteps  
time = np.arange(0,Nt*h-0.0001,h)

