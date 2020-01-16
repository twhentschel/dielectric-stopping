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
T    = 6     # eV
ne   = 1e23  # e/cc
muau = 0.126 # au

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

# # Make the data
# k = 10 # a.u.
# w = np.linspace(0, 150*wp, 200)
# elf = np.asarray([MD.ELF(k, x, Tau, muau, nu) for x in w])

# # Let's plot!
# plt.plot(w/wp, elf, label='k = {} au'.format(k), c='C1')
# plt.title(r'Al z* = 3; $T$ = {} eV, $n_e$ = {} e/cc'.format(T, ne))
# plt.xlabel(r'$\omega/\omega_p$')
# plt.ylabel('ELF')
# plt.legend()

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
# #plt.plot(omegaintegral.t/wp, omegaintegral.y[0])
# plt.show()

####### Step 3: Omega integral code test #######

# # Define the integrand
# omegaintegrand = lambda w, y : w * MD.ELF(k, w, Tau, muau, nu)

# def modBG_wp(n, k, kbT):
#     """
#     Modified Bohm-Gross dispersion relation, given in Glezner & Redmer, Rev.
#     Mod. Phys., 2009, Eqn (16).
    
#     n - electron density (au)
#     k - wavenumber (au)
#     kbT - thermal energy (au)
#     """  
#     wp    = np.sqrt(4*np.pi*n)
#     BG_wp = np.sqrt(wp**2 + 3*kbT*k**2)
#     thermal_deBroglie = np.sqrt(2*np.pi/kbT)
    
#     return np.sqrt(BG_wp**2 + 3*kbT*k**2 \
#                     * 0.088 * n * thermal_deBroglie**3 \
#                     + (k**2/2)**2)

# v = 30

# # Find minimum of the real part of the dielectric
# reeps = lambda x : MD.MerminDielectric(k, x, Tau, muau, nu).real
# omegamin = opt.minimize_scalar(reeps, bracket=(0, modBG_wp(neau, k, Tau)),
#                                method='brent')

# root = modBG_wp(neau, k, Tau)
# if reeps(omegamin.x) < 0:
#     root = opt.newton(reeps, root)

# omegaintegral = 0
# # Set the integral tolerance
# tol = 1.49012e-6

# if k*v <= root:
#     # Solve the corresponding ODE
#     y0 = [0]
#     wlim = (0, k*v)
#     omegaintegral = solve_ivp(omegaintegrand, wlim, y0, 
#                               rtol=tol, atol=tol).y[0][-1]
# else:
#     # Determine the effective infinity of the ELF - the omega at which the ELF
#     # is roughly zero (ELF < tol)
#     effinf = root + np.sqrt(2*(10*Tau + muau))*k

#     if k*v > effinf:
#         omegaintegral = 0
#     else:
#         y0 = [0]
#         wlim = (effinf, k*v)

#         omegaintegral = sumrule + solve_ivp(omegaintegrand, wlim, y0,
#                                             rtol=tol, atol=tol).y[0][-1]

# print("kv = {}".format(k*v))
# print("wp = {}".format(wp))
# print("wmax = {}".format(root))

# #plt.scatter(omegaintegral.t/wp, np.zeros(len(omegaintegral.t)), marker='+')
# plt.show()


####### Step 4: integral over k attempt #######
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

def omegaintegral(v, k, T, mu, nu):
    # print(k)
    # Get this case out of the way!
    if k == 0:
        return 0

    # Define the integrand
    omegaintegrand = lambda w, y : w * MD.ELF(k, w, T, mu, nu)/k

    # Find minimum of the real part of the dielectric
    reeps = lambda x : MD.MerminDielectric(k, x, T, mu, nu).real
    omegamin = opt.minimize_scalar(reeps, bracket=(0, modBG_wp(neau, k, T)),
                                   method='brent')
    
    root = modBG_wp(neau, k, T)
    if reeps(omegamin.x) < 0:
        try:
            # At low-k, newton's method breaks
            root = opt.newton(reeps, root)
        except RuntimeError:
            # print("Switching from Newton's method to the safer bisection " + 
            #       "method to find the maximum position of the ELF.")
            maxiter = 10
            upperlim = root
            while reeps(upperlim) < 0 and maxiter > 0:
                upperlim = upperlim + k
            root = opt.brentq(reeps, omegamin.x, upperlim)

    omegaintegral = 0
    # Set the integral tolerance
    tol = 1.49012e-6
    
    if k*v <= root:
        # Solve the corresponding ODE
        y0 = [0]
        wlim = (0, k*v)
        omegaintegral = solve_ivp(omegaintegrand, wlim, y0, 
                                  rtol=tol, atol=tol).y[0][-1]
    else:
        # Determine the effective infinity of the ELF - the omega at which the 
        #ELF is roughly zero (ELF < tol)
        effinf = root + np.sqrt(2*(10*T + mu))*k

        if k*v > effinf:
            return 0

        y0 = [0]
        wlim = (effinf, k*v)

        omegaintegral = sumrule + solve_ivp(omegaintegrand, wlim, y0,
                                            rtol=tol, atol=tol).y[0][-1]

    return omegaintegral

def momintegral(v, T, mu, nu):
    # Define the integrand
    momintegrand = lambda k, y: omegaintegral(v, k, T, mu, nu)
    
    y0 = [0]
    klim = (0, 2*v)
    first_step = 0.01
    if v < 0.01:
        first_step = None
    momintegral = solve_ivp(momintegrand, klim, y0, first_step=first_step)
    
    return momintegral.y[0][-1]

# v = np.linspace(0, 30)
# import time

# kint = []
# start = time.time()
# for x in v:
#     s = time.time()
#     kint.append(momintegral(x, Tau, muau, nu))
#     e = time.time()
#     print("v = {} run: {} s".format(x, e-s))
# end = time.time()
# print("total time = {} for {} velocity points".format(end-start, len(kint)))
# plt.plot(v, kint, label='RPA')
# plt.title("T = {}, ne = {}".format(Tau, muau))
# plt.xlabel("Velocity (au)")
# plt.ylabel("Stopping power integral")
# #print("Stopping power integral (v = {}) = {}".format(v, kint))

# plt.savefig("stoppingpower_1.png")
# plt.show()

v=[1, 2, 3, 10]
kpoints = np.linspace(0, 10)
for q in v:
    omegaint = [omegaintegral(q, x, Tau, muau, nu) for x in kpoints]
    plt.plot(kpoints, omegaint, label="v = {}".format(q))
plt.xlabel("k")
plt.ylabel("k-integrand")
plt.legend()
plt.show()
