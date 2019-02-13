import numpy as np
import matplotlib.pyplot as plt

from anim import make_3d_animation

N = 3
L = 10
Nt = 200

pos = np.zeros(shape=(Nt, N, 3), dtype=float)
pos[0] = np.random.rand(N, 3) * L
vel = np.random.rand(N, 3) - 0.5

for t in range(1, Nt):
    pos[t] = (pos[t-1] + 0.2 * vel) % L

anim = make_3d_animation(L, pos, delay=30, rotate_on_play=0)
plt.show()
