import numpy as np
import matplotlib.pyplot as plt

N = 3
L = 10
Nt = 200

pos = np.zeros(shape=(Nt, N, 2), dtype=float)
pos[0] = np.random.rand(N, 2) * L
vel = np.random.rand(N, 2) - 0.5

for t in range(1, Nt):
    pos[t] = (pos[t-1] + 0.2 * vel) % L

print(pos[t])
