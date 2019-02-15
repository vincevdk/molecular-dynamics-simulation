import numpy as np
import matplotlib.pyplot as plt

from anim import make_3d_animation

N = 3
L = 10 
Nt = 200
m = 1
h = 1/(Nt)



vel = np.zeros([Nt,2])
pos = np.zeros([Nt,2])

F = 2*pos

x = np.random.randint(0,Nt)

vel[0,0] = np.random.rand()
pos[0,0] = np.random.rand()



for v in range(1,Nt):
    vel[v] = vel[v-1]+(1/m)*h
    print(vel)


for t in range(1,Nt): 
    pos[t] = (pos[t-1]+h*v[t-1]) % L

