#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figuring out how to compute the omega integral when calculating the stopping 
power.

@author: tommy
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp 
import scipy.optimize as opt
# User defined pacakges
import dielectricfunction.MerminDielectric as MD


# Define some fixed parameters

# Need to input these manually from an external source - the first three are 
# all related.
T    = 1     # eV
ne   = 1e23  # e/cc
muau = 0.279 # au

nu = 0
# Convert to a.u. if necessary
ev2au = 1/27.2114 #a.u./eV
a0 = 0.529*10**-8 # Bohr radius, cm
cm2au = 1/a0
Tau   = T  * ev2au
neau  = ne * 1/cm2au**3

# Important constants
kF = np.sqrt((3 * np.pi**2 * neau)**(2/3))
wp = np.sqrt(4 * np.pi * neau)
sumrule = np.pi/2 * wp**2


####### Step 1: Take a peak #######

# Make the data
k = 0.2 # a.u.
w = np.linspace(0, 4*wp, 200)
elf = np.asarray([MD.ELF(k, x, Tau, muau, nu) for x in w])

# Let's plot!
plt.plot(w/wp, elf, label='k = {} au'.format(k), c='C1')
plt.title(r'Al z* = 3; $T$ = {} eV, $n_e$ = {} e/cc'.format(T, ne))
plt.xlabel(r'$\omega/\omega_p$')
plt.ylabel('ELF')
plt.legend()

#plt.show()

####### Step 2: Integration attempt 1 #######
# '''
# For this step, we want to compare the integration of the ELF with the 
# corresponding sum rule to check for correctness.

# We will just eye-ball the upper limit of the integral using the graphs from
# step 1.
# '''

# k = k

# # Define the integrand
# omegaintegrand = lambda w, y : w * MD.ELF(k, w, Tau, muau, nu)

# # Set the initial value
# y0 = [0]
# # Set the integration limits
# wlim = (0, 4*wp)

# # Set the tolerance
# tol = 1.49012e-6
# # Solve the corresponding ODE
# omegaintegral = solve_ivp(omegaintegrand, wlim, y0, #method='LSODA',
#                           rtol=tol, atol=tol)

# print("omega integral = {} ; sum rule = {}".format(omegaintegral.y[0][-1],
#                                                    sumrule))
# plt.scatter(omegaintegral.t/wp, np.zeros(len(omegaintegral.t)), marker='+')
# plt.plot(omegaintegral.t/wp, omegaintegral.y[0])
# plt.show()

####### Step 3: Omega integral code test #######

# Define the integrand
omegaintegrand = lambda w, y : w * MD.ELF(k, w, Tau, muau, nu)

def modBG_wp(n, k, kbT):
    """
    Modified Bohm-Gross dispersion relation, given in Glezner & Redmer, Rev.
    Mod. Phys., 2009, Eqn (16).
    
    n - electron density (au)
    k - wavenumber (au)
    kbT - thermal energy (au)
    """  
    wp    = np.sqrt(4*np.pi*n)
    BG_wp = np.sqrt(wp**2 + 3*kbT*k**2)
    thermal_deBroglie = np.sqrt(2*np.pi/kbT)
    
    return np.sqrt(BG_wp**2 + 3*kbT*k**2 \
                    * 0.088 * n * thermal_deBroglie**3 \
                    + (k**2/2)**2)

v = 1

# Find minimum of the real part of the dielectric
reeps = lambda x : MD.MerminDielectric(k, x, Tau, muau, nu).real
omegamin = opt.minimize_scalar(reeps, bracket=(0, modBG_wp(neau, k, Tau)),
                               method='brent')

root = modBG_wp(neau, k, Tau)
if reeps(omegamin.x) < 0:
    root = opt.newton(reeps, root)

omegaintegral = 0
# Set the integral tolerance
tol = 1.49012e-6

if k*v <= root:
    # Solve the corresponding ODE
    y0 = [0]
    wlim = (0, k*v)
    omegaintegral = solve_ivp(omegaintegrand, wlim, y0, #method='LSODA',
                              rtol=tol, atol=tol)
else:
    # Determine the effective infinity of the ELF - the omega at which the ELF
    # is roughly zero (ELF < tol)
    effinf = 12*wp
    y0 = [0]
    wlim = (effinf, k*v)

    omegaintegral = sumrule + solve_ivp(omegaintegrand, wlim, y0, 
                                        rtol=tol, atol=tol).y[0][-1]

print("kv = {}".format(k*v))
print("wp = {}".format(wp))
print("wmax = {}".format(root))

plt.scatter(omegaintegral.t/wp, np.zeros(len(omegaintegral.t)), marker='+')
plt.show()
