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
from scipy.integrate import quad
# User defined pacakges
import dielectricfunction.MerminDielectric as MD



# Define some fixed parameters

# Need to input these manually from an external source - these three are all 
# related.
T    = 6     # eV
ne   = 1e23  # e/cc
muau = 0.126 # au

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
k = 0.4 # a.u.
w = np.linspace(0, 4*wp, 200)
elf = np.asarray([MD.ELF(k, x, Tau, muau, 0) for x in w])

# Let's plot!
plt.plot(w/wp, elf, label='k = {} au'.format(k))
plt.title(r'Al z* = 3; $T$ = {} eV, $n_e$ = {} e/cc'.format(T, ne))
plt.xlabel(r'$\omega/\omega_p$')
plt.ylabel('ELF')
plt.legend()

#plt.show()

####### Step 2: Integration attempt 1 #######
'''
For this step, we want to compare the integration of the ELF with the 
corresponding sum rule to check for correctness.

We will just eye-ball the upper limit of the integral using the graphs from
step 1.
'''

k = k

# Define the integrand
omegaintegrand = lambda w, y : w * MD.ELF(k, w, Tau, muau, 0)

# Set the initial value
y0 = [0]
# Set the integration limits
wlim = (0, 4*wp)

# Set the tolerance
tol = 1.49012e-8
# Solve the corresponding ODE
omegaintegral = solve_ivp(omegaintegrand, wlim, y0,# method='DOP853',
                          rtol=tol, atol=tol)

print("omega integral = {} ; sum rule = {}".format(omegaintegral.y[0][-1],
                                                   sumrule))
plt.scatter(omegaintegral.t/wp, np.zeros(len(omegaintegral.t)))

plt.show()
