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
from scipy.integrate import solve_ivp
from scipy import optimize
import time

from Mermin import MerminDielectric as MD
from xMermin import xMermin as xmd

# for PIMC UEG data
from keras import losses
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU

######## Parameters ###############
a0 = 0.529*10**-8 # Bohr radius, cm
TeV = 10 # Temperature, eV
ne_cgs = 6*10**23 # electron density, cm^-3
neau = ne_cgs * a0**3 # electron density, au
EFau =  0.5*(3*np.pi**2*neau)**(2/3)# Fermi energy, au
kFau = np.sqrt(2*EFau) # Fermi momentum/wavevector, au
Tau = TeV/27.2114 # au
#kFau = kF_eV / 27.2114 # au
wpau = np.sqrt(4*np.pi*neau)
sumrule = np.pi/2 * wpau**2

# Needed to assume some mu. For this, we are assuming the material and then
# using a spreadsheet that Stephanie prepared to get mu
muau = 0.783 # mu for Hydrogen at 10 eV, 10g/cc

#########################################

###### Collision frequencies data ########
filename = "tests/H_10gpcc_10_eV_vw.txt"
wdata, RenuT, ImnuT = np.loadtxt(filename, unpack=True)

wmin = min(wdata)
#nu = 1j*ImnuT; nu += RenuT

# Parametrize
f_renu = interpolate.interp1d(wdata, RenuT)
f_imnu = interpolate.interp1d(wdata, ImnuT)

nu = lambda w: f_renu(w) + 1j*f_imnu(w)
#nu = lambda w: neau * RenuT[0]/(RenuT[0]**2 + w**2)

# def nu(w): 
#     return 1e-2 # Don't have collisions !
        
##### Use a simple expression for collision frequencies #####
# def nu(w):
#     return RenuT[0] / (1 + w**2)
##### Omega integral #####
# w = np.linspace(wmin, 10, 200)
# k = 1e-1
######################################################

#########################################

"""
Obtain static local field correction from PIMC data of uniform electron gas.
"""
# Define LFC
LR = LeakyReLU()
LR.__name__ = 'relu'

# Define the Keras model

N_LAYER = 40
W_LAYER = 64

model = Sequential()
model.add(Dense(W_LAYER, input_dim=3, activation=LR))

REGULARIZATION_RATE = 0.0000006

for i in range(N_LAYER-1):
	model.add( Dense( W_LAYER, activation=LR, 
                  kernel_regularizer=regularizers.l2( REGULARIZATION_RATE)))

model.add(Dense(1, activation='linear'))

# Load the trained weights from hdf5 file
model.load_weights('../dielectricfunction/LFC/tests/LFC.h5')

# Define simple wrapper function (x=q/q_F):

def GPIMC(q, ne, T):
    rs = (3/4/np.pi/ne)**(1./3)
    kF = (9.0*np.pi/4.0)**(1.0/3.0)/rs
    EF = 0.5*(3*np.pi**2*ne)**(2./3)
    x = q/kF
    y = T/EF
    result = model.predict( np.array( [[x,rs,y]] ) )
    return result[0][0]

########################################################

# # Define the integrand
# omegaintegrand = lambda x, y : x * MD.ELF(k, x, nu(x), Tau, muau)

# # integrate
# s = time.time()
# integrand = [omegaintegrand(x, 0) for x in w]
# I1 = integrate.trapz(integrand, w)
# e = time.time()
# print("trapezoidal integration = {}, time ={} s".format(I1, e-s))
# s = time.time()
# I2 = integrate.quad(omegaintegrand, wmin, 10, args=(0), epsabs=1e-3, epsrel=1e-3)
# e = time.time()
# print("adaptive integration = {}, time = {} s".format(I2, e-s))
# s= time.time()
# I3 = solve_ivp(omegaintegrand, (wmin, 10), [0])
# e = time.time()
# print("ODE solve = {}, time = {} s".format(I3.y[0][-1], e-s))
# print("Sum Rule = {}".format(sumrule))


# plt.plot(w, integrand)
# plt.show()

# k-integrand
def omegaint(k, v, nu, T, mu):
    
    omegaintegrand = lambda x : x / k * MD.ELF(k, x, nu(x), T, mu) 
    
    if k*v < wmin :
        return 0
    if k*v > max(wdata):
        return 0
    w = np.linspace(wmin, k*v, 100)
    
    integrand = [omegaintegrand(x) for x in w]
    
    return integrate.trapz(integrand, w)

def omegaint_adapt(k, v, nu, T, mu, G):
    
    omegaintegrand = lambda x, y : x * xmd.ELF(k, x, nu(x), T, mu, 
                                               G(k, neau, Tau)) 
    
    if k*v < wmin :
        return 0
    if k*v > max(wdata):
        return 0
    I = solve_ivp(omegaintegrand, (wmin, k*v), [0], rtol=1e-6, 
                  atol=1e-6)

    return I.y[0][-1]

### Look at the k-integrand ###
# v = [7]
# k = np.linspace(1e-3, 0.5, 100)5B
# fig, ax = plt.subplots()
# for y in v:
#     kint = [omegaint_adapt(x, y, nu, Tau, muau) for x in k]
#     p = ax.plot(k, kint, label='v={}'.format(y))
#     #ax.axvline(2*y, linestyle='-.', c=p[-1].get_color())

# ax.set_xlabel(r'$k (1/a_0)$')
# ax.set_title('k-integrand for Al at T=6 eV')
# ax.legend()

# #plt.savefig('kintegrand2')
# plt.show()


def momint(v, nu, T, mu, k0):
    
    # Define the integrand
    #momintegrand = lambda k: omegaint(v, k, nu, T, mu)
    k = np.linspace(k0, 2*v, 100)
    integrand = [1/ x * omegaint(x, v, nu, Tau, muau) for x in k]
    
    return integrate.trapz(integrand, k)

def momint_adapt(v, nu, T, mu, G, k0):
    kintegrand = lambda k, y : 1/k * omegaint_adapt(k, v, nu, T, mu, G)
    sol = solve_ivp(kintegrand, (k0, 2*v), [0])
    return sol.y[0][-1]

def drude_ELF(w, nu, wp):
    drude_eps = 1 - wp**2 / (w**2 + 1j * w * nu(w))
    return drude_eps.imag / (drude_eps.real**2 + drude_eps.imag**2)


def error1(v, nu, wmax, wp, k0):
    
    return v * k0 * wmax * drude_ELF(wmax, nu, wp)

def error2(v, nu, T, mu, k0):
    return v**2 * k0**2 / 2 * MD.ELF(k0, k0*v, nu(k0*v), T, mu)


v = np.linspace(1e-3, 12, 100)
k0 = 5e-2
s = time.time()
S = [momint_adapt(x, nu, Tau, muau, GPIMC, 5e-2) for x in v]
runtime = 'run time = {} s\n'.format(time.time() - s)
parameters = 'Te = {} [eV]\nne = {:e} [1/cc]\nmu = {} [au]\n'.\
             format(TeV, ne_cgs,muau)
head = 'v[a.u.]    stopping number[a.u.]'
S = np.asarray(S)
# Save data just in case something breaks
np.savetxt('stopdata_xmermin_hydrogen_1.txt', np.transpose([v, S]), 
           header = runtime + parameters + head)

# # S = np.loadtxt('stopping_data_adapt_tmp.out') / kFau**2

# wmax_drude = 0.588
# S_error1 = [2 / np.pi / x**2 * error1(x, nu, wmax_drude, wpau, 5e-2) 
#             for x in v]
# S_error1 = np.asarray(S_error1) / kFau**2

# # S_error2 will break if k0*v is less than wmin
# S_error2 = [2 / np.pi / x**2 * error2(x, nu, Tau, muau, 5e-2) 
#             for x in v]
# S_error2 = np.asarray(S_error2) / kFau**2

# plt.figure()
# #plt.fill_between(v[1:], S[1:], S[1:] + S_error1[1:], color='C2', alpha=0.2)
# plt.fill_between(v, S, S + S_error2, color = 'C1',  alpha=0.2)
# plt.xlabel(r'$v$ (au)')
# plt.ylabel(r'$S(v) / Z^2$')
# plt.title('Stopping power for solid Al at 6 ev')

# plt.savefig('stoppingpower_drudenu2')

