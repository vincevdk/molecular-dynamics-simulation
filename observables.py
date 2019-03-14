import numpy as np
import numpy.ma as ma
from config import *

def calculate_pressure(vir):
    p = N_particle*kb*temperature/(L*L)-(1/3)*0.5*vir
    return(p)


