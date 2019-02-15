import numpy as np
import matplotlib.pyplot as plt

# import velocity of particles

N = 3
L = 10
Nt = 200
m = 1
h = 1/(Nt)
massa = 3
vel = np.zeros([Nt,2])
pos = np.zeros([Nt,2])

F = 2*pos

x = np.random.randint(0,Nt)

vel[0,0] = np.random.rand()
pos[0,0] = np.random.rand()



for v in range(1,Nt):
    vel[v] = vel[v-1]+(1/m)*h


for t in range(1,Nt):
    pos[t] = (pos[t-1]+h*vel[t-1]) % L

 
speed = (vel[:,0]**2 +  vel[:,1]**2)**0.5
kinetic_energy = 0.5 * speed * massa
print(speed)

if __name__ == "__main__":
    x_axis = np.linspace(1,Nt,Nt)
    plt.plot(x_axis, speed)
    plt.plot(x_axis, kinetic_energy)
#    plt.legend()
plt.show()
