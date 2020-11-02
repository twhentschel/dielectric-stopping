#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 17:29:10 2020

@author: tommy

Calculates the stopping power based on the dielectric formalism, 
see: M. D. Barriga-Carrasco, PRE, 79, 027401 (2009)
"""
import numpy as np
from scipy.integrate import solve_ivp

def stopnumber(v, elf, elfargs, wmin=0, kmin=0, wtol=(1e-6,1e-6), vlow=0):
    '''
    Calculates the integral portion over momentum and energy space of the
    stopping power directly (i.e. no funny business), which, I think, is called
    the stopping number.

    Parameters
    ----------
    v : float
        Initial particle velocity for the stopping number calculation.
    elf : function
        The electron loss function we intend to use. The form of the arguments
        should be: elf(k, omega, *elfargs), where k is the momentum variable
        and omega is the energy variable.
    elfargs : tuple
        Material inputs that go into the dielectric function, like temperautre,
        chemical potential, etc. If the ELF depends on parameters that have
        some energy dependence (like collision frequency), or momentum
        dependence (like local field corrections), those should be folded into
        the elf function before passing it into this function.
    wmin, kmin : float, optional
        Sometimes, I find that the elf functions are quite convoluted and do
        not work for really small energies (omega/w) or momenta (k). These two
        parameters help stop our integration from treading into those uncertain
        territories.
    wtol : list-like, optional
        A 2-element list that holds the relative and absolute tolerances used 
        in the omega/energy integration, respectively.
    vlow : scalar, optional
        This parameter is useful for the sequentialstopping function. It
        represents the lower integration limit for the omega integral.

    Returns
    -------
    L : float
        The stopping number for a given initial particle velocity, v.
    '''
    
    # Energy (inner) integral part
    def omegaint(k):
        # Assignment (a few lines down) make vlow a local variable, need to
        # declare it first.
        nonlocal vlow
        
        # integrand
        omegaintegrand = lambda x, y : x/k * elf(k, x, *elfargs)
        # prevent integration limits from going too low
        if k*v <= wmin:
            return 0
        if k*vlow <= wmin:
            # k == 0 is handled by above if statement
            vlow = wmin / k
        # Do the integral, which I actually turn into an ODE problem and use
        # an ODE solver with adaptive step sizes:
        # dy/dt = f(t, y); y(0) = y0
        # y = solve_ivp(f, (lower t limit, upper t limit), y0,
        #                   relative tolerance, absolute tolerance)
        I = solve_ivp(omegaintegrand, (k*vlow, k*v), [0], rtol=wtol[0], 
                      atol=wtol[1])
        # I.y[0][-1] gives us the answer
        return I.y[0][-1]

    # Momentum (outer) integral part
    kintegrand = lambda k, y : omegaint(k)
    # If I need to call out specific params in elfargs, then it is pointless to
    # use elfargs for generalty!!
    T, mu = elfargs
    kwidth = (2 * (10 * T + mu))**(0.5)
    I = solve_ivp(kintegrand, (kmin, 2*(v + kwidth)), [0])
    L = I.y[0][-1]
    return L

def stopnumberArr(v, elf):
    '''
    Calculates the integral portion over momentum and energy space of the
    stopping power, when given a 2D array of the ELF spanning momentum and
    energy space.

    Parameters
    ----------
    v : float
        Initial velocity for the stopping number calculation.
    elf : array-like
        A 2D-array of precomputed values for the electron loss function (ELF)
        that will be used to integrate the ELF over momentum and energy space. 

    Returns
    -------
    L : float
        The stopping number for a given initial velocity, v.
    '''
    return 0

def sequentialstopping(varr, elf, T, mu, wmin, kmin, L0=0):
    '''
    varr: array-like
        Array of velocities. Assumption is that the first velocity corresponds
        zero stopping, but you can control this with L0.
    '''
    L = np.zeros(len(varr))
    L[0] = L0
    for i in range(1, len(varr)):
        L[i] = stopnumber(varr[i], elf, elfargs=(T,mu), wmin=wmin, kmin=kmin,
                          vlow=varr[i-1])

    return np.cumsum(L)
