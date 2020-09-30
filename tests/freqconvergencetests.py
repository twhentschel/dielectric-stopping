#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 16:54:24 2020

@author: tommy

The RPA dielectric stopping power is hard - I cannot do it.
I can do the Mermin one though. Maybe there is a finite collision frequency
that is small enough that I can approximate the RPA, but still have the ELF not
be so sharp.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from contextlib import contextmanager
import os

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

dielfuncdir = '~/Documents/Research/WDM_codes/GitProjects/dielectricfunction/'
'''
with does not create a scope.
'''
with cd(dielfuncdir):
    import Mermin.MerminDielectric as md
    import LFC.LFCDielectric as lfc
    import xMermin.xMermin as xmd


from dielectricstopping import stopnumber

# for PIMC UEG data
from keras import losses
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU

######## Parameters ###############
a0 = 0.529*10**-8 # Bohr radius, cm
TeV = 10 # Temperature, eV
ne_cgs = 6*10**24 # electron density, cm^-3
neau = ne_cgs * a0**3 # electron density, au
EFau =  0.5*(3*np.pi**2*neau)**(2/3)# Fermi energy, au
kFau = np.sqrt(2*EFau) # Fermi momentum/wavevector, au
Eh = 27.2114 # eV/au
Tau = TeV/Eh # au
#kFau = kF_eV / 27.2114 # au
wpau = np.sqrt(4*np.pi*neau)
sumrule = np.pi/2 * wpau**2

# Needed to assume some mu. For this, we are assuming the material and then
# using a spreadsheet that Stephanie prepared to get mu
muau = 4.35


auvelocity2eV = lambda vau : 2.5e4 * vau**2 # [eV]
austopping2SI = Eh/a0 # [eV/cm]
#########################################
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

##########################################

###### Collision frequencies data ########
filename = "tests/H_10gpcc_10_eV_vw.txt"
wdata, RenuT, ImnuT = np.loadtxt(filename, unpack=True)

wmin = min(wdata)
#nu = 1j*ImnuT; nu += RenuT

# Parametrize
f_renu = interpolate.interp1d(wdata, RenuT)
f_imnu = interpolate.interp1d(wdata, ImnuT)

nu = lambda w: f_renu(w) + 1j*f_imnu(w)
##########################################


# Set up our ELF for use in integration code, since nu and GPIMC are functions
save = []
for i in range(1):
    factor = 10**i
    elf = lambda k, w, T, mu: xmd.ELF(k, w, nu(w)/factor, T, mu,
                                      GPIMC(k, neau, T))
    # Define the stopping power function in terms of a velocity
    Z = 1
    k0 = 5e-2
    S = lambda v : 2 * Z**2 / np.pi / v**2 \
        * stopnumber(v, elf, (Tau, muau), wmin=wmin, kmin=k0)
    v = np.asarray([1.24345, 2.2426, 3, 4.41081])
    Svals = np.asarray([S(x) for x in v])
    save.append(Svals)
    plt.scatter(v, Svals, label='nu/{:.1e}'.format(factor))

vau, Lmerm = np.loadtxt('stoppingdata/stopdata_mermin_carbon_1.out', unpack=True)
Smerm = 2 * 1 /np.pi / vau**2 * Lmerm
plt.plot(vau, Smerm)
#plt.legend()
# Define the stopping power function in terms of a velocity

# L = lambda v : stopnumber(v, elf, (Tau, muau), wmin=wmin, kmin=k0)

