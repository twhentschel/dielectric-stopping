#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 08:50:07 2020

@author: tommy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.integrate as integrate
from scipy import optimize

from dielectricfunction_symln.Mermin import MerminDielectric as MD

######## Parameters ###############
a0 = 0.529*10**-8 # Bohr radius, cm
TeV = 6 # Temperature, eV
ne_cgs = 1.8*10**23 # electron density, cm^-3
neau = ne_cgs * a0**3 # electron density, au
EFau =  0.5*(3*np.pi**2*neau)**(2/3)# Fermi energy, au
kFau = np.sqrt(2*EFau) # Fermi momentum/wavevector, au
Tau = TeV/27.2114 # au
#kFau = kF_eV / 27.2114 # au
wpau = np.sqrt(4*np.pi*neau)
sumrule = np.pi/2 * wpau**2

# Needed to assume some mu. For this, we are assuming the material and then
# using a spreadsheet that Stephanie prepared to get mu
muau = 0.305 # mu for aluminum, with ne_cgs=1.8*10**23, T=1ev, Z*=3; au

#########################################

###### Collision frequencies data ########
filename = "tests/Al_6_eV_vw.txt"
wdata, RenuT, RenuB, ImnuT, ImnuB = np.loadtxt(filename, skiprows = 1, 
                                               unpack=True)

wmin = min(wdata)
nu = 1j*ImnuT; nu += RenuT

# Parametrize
f_renu = interpolate.interp1d(wdata, RenuT)
f_imnu = interpolate.interp1d(wdata, ImnuT)

nu = lambda w: f_renu(w) + 1j*f_imnu(w)
#nu = lambda w: neau * RenuT[0]/(RenuT[0]**2 + w**2)

# w = np.linspace(wmin, 100, 200)
# k = 1

# # Define the integrand
# omegaintegrand = lambda x : x * MD.ELF(k, x, nu(x), Tau, muau)

# integrand = [omegaintegrand(x) for x in w]

# # integrate
# I1 = integrate.trapz(integrand, w)
# #I2 = integrate.quad(omegaintegrand, wmin, 2)
# print("trapezoidal integration = {}".format(I1))
# #print("adaptive integration = {}".format(I2))
# print("Sum Rule = {}".format(sumrule))


# #eps= np.asarray([MD.MerminDielectric(k, x, nu(x), Tau, muau) for x in w])

# #plt.plot(w, eps.real, label='real')
# #plt.plot(w, eps.imag, label='imag')

# #plt.plot(w, eps.imag/(eps.imag**2 + eps.real**2))
# #plt.legend()
# #plt.show()

# k-integrand
def omegaint(v, k, nu, T, mu):
    
    omegaintegrand = lambda x : x / k * MD.ELF(k, x, nu(x), T, mu) 
    
    if k*v < wmin :
        return 0
    if k*v > max(wdata):
        return 0
    w = np.linspace(wmin, k*v, 100)
    
    integrand = [omegaintegrand(x) for x in w]
    
    return integrate.trapz(integrand, w)

### Look at the k-integrand ###
# v = [0.1, 1, 5, 10]
# k = np.linspace(5e-2, 5, 2)
# fig, ax = plt.subplots()
# for y in v:
#     kint = [omegaint(y, x, nu, Tau, muau) for x in k]
#     p = ax.plot(k, kint, label='v={}'.format(y))
#     ax.axvline(2*y, linestyle='-.', c=p[-1].get_color())

# ax.set_xlabel(r'$k ~ (1/a_0)$')
# ax.set_title('k-integrand for Al at T=6 eV')
# ax.legend()

# plt.show()

def momint(v, nu, T, mu, k0):
    
    # Define the integrand
    #momintegrand = lambda k: omegaint(v, k, nu, T, mu)
    k = np.linspace(k0, 2*v, 100)
    integrand = [omegaint(v, x, nu, Tau, muau) for x in k]
    
    return integrate.trapz(integrand, k)

def drude_ELF(w, nu, wp):
    drude_eps = 1 - wp**2 / (w**2 + 1j * w * nu(w))
    return drude_eps.imag / (drude_eps.real**2 + drude_eps.imag**2)


def error(v, nu, wmax, wp, k0):
    # find maximum of the drude ELF
    return v * k0 * wmax * drude_ELF(wmax, nu, wp)

v = np.linspace(1e-3, 10, 2)
S = [2 / np.pi / x**2 * momint(x, nu, Tau, muau, 5e-2) for x in v]
S = np.asarray(S)
# Save data just in case something breaks
np.save('stopping_data_tmp', S)

wmax_drude = 0.588
S_error = [2 / np.pi / x**2 * error(x, nu, wmax_drude, wpau, 5e-2) for x in v]
S_error = np.asarray(S_error)
#prefactor = 2 / np.pi / v**2
plt.plot(v, S)
plt.fill_between(v, S, S + S_error, alpha=0.2)

plt.xlabel('v (au)')
plt.ylabel('Stopping Power (au)')
plt.title('Stopping power for solid Al at 6 ev')



