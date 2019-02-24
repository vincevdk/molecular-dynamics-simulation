import numpy as np

# Physical values are defined globally

#good value for the time step is 10−14 s which in units of (ms 2/e )1/2 is equal to about 0.004.

m = 6.6 * 10**-26
dim = 3
temperature = 230
N_particle = 10
sigma = 3.405*10**-10
kb = 1.38*10**-23
epsilon = 119.8*kb
L = 1000

timestep = 10*10**-14
dimensionless_timestep = timestep/(m*sigma**2/epsilon)**.5
Nt = 200 # number of timesteps  
time = np.arange(0,Nt*dimensionless_timestep-0.04,dimensionless_timestep)

