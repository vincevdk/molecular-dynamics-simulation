import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#read datafile
df = pd.read_csv('data_final2.csv')
pres_sim = df.pressure
u_sim = df.pot_energy
pres_err = df.P_error
u_err = df.U_error
density = df.density
T_set = df.T_set
T_final = df.T_final
x = np.arange(len(u_verlet))

#plots for different obeservables

plt.figure()
plt.plot(x,p_verlet,label='Verlet')
plt.plot(x,pres_sim,label='simulation')
plt.title(r'Compressibility factor') 
plt.xlabel('simulation number')
plt.ylabel(r'$\beta$P/$\rho$')
plt.legend(loc='best')
plt.savefig('presplot')

plt.figure()
plt.plot(x,u_verlet,label='Verlet')
plt.plot(x,u_sim,label='simulation')
plt.title('Potential energy') 
plt.xlabel('simulation number')
plt.ylabel('energy ($\epsilon$)')
plt.legend(loc='best')
plt.legend(loc='best')
plt.savefig('potplot')




