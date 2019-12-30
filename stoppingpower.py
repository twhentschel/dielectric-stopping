#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 10:04:32 2019

@author: tommy
"""

import numpy as np
import time
from scipy import integrate
from dielectricfunction.Mermin import MerminDielectric as MD

# Plasma frequency, a.u.
def w_p(n):
    return np.sqrt(4 * np.pi * n )

"""
Calculate L, and use some number as infinity.
"""


def stoppower(T, mu, n, nu, v):
    def omegaint(k):
        if k == 0:
            return 0 
        
        def integrand(w):
            return w*MD.ELF(w, k, T, mu, nu)
            
        return 2*integrate.quad(integrand, 0, k*v)[0]/k

    return 1/(np.pi * w_p(n)**2 ) \
            * integrate.quad(omegaint, 0, 2*v)[0]           
       

import matplotlib.pyplot as plt

######## Parameters ###############
a0 = 0.529*10**-8 # Bohr radius, cm
TeV = 6 # Temperature, eV
ne_cgs = 1.8*10**23 # electron density, cm^-3
neau = ne_cgs * a0**3 # electron density, au
EFau =  0.5*(3*np.pi**2*neau)**(2/3)# Fermi energy, au
kFau = np.sqrt(2*EFau) # Fermi momentum/wavevector, au
T_au = TeV/27.2114 # au
#kFau = kF_eV / 27.2114 # au
wpau = np.sqrt(4*np.pi*neau)
vth = np.sqrt(T_au)
# Needed to assume some mu. For this, we are assuming the material and then
# using a spreadsheet that Stephanie prepared to get mu
muau = 0.305 # mu for aluminum, with ne_cgs=1.8*10**23, T=6ev, Z*=3; au



#########################################


filename = "dielectricfunction/tests/Al_6_eV_vw.txt"
w, RenuT, RenuB, ImnuT, ImnuB = np.loadtxt(filename, skiprows = 1, unpack=True)
nu = 1j*ImnuT; nu += RenuT

v =  vth * np.linspace(0, 30, 2)
s = [stoppower(T_au, muau, neau, y, x) for x,y in zip(v,nu)]
plt.plot(v/vth, s)
plt.ylabel("Stopping Power")
plt.xlabel("v/vth")
plt.xlim(0, 30)
plt.ylim(0, 0.4)
plt.show()

