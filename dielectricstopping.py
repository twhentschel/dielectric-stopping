#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 17:29:10 2020

@author: tommy

Calculates the stopping power based on the dielectric formalism, 
see: M. D. Barriga-Carrasco, PRE, 79, 027401 (2009)
"""

from scipy.integrate import solve_ivp

def stopnumber(v, elf, elfargs, wmin=0, kmin=0, wtol=(1e-6,1e-6)):
    '''
    Calculates the integral portion over momentum and energy space of the
    stopping power directly (i.e. no funny business), which, I think, is called
    the stopping number.

    Parameters
    ----------
    v : float
        Initial velocity for the stopping number calculation.
    elf : function
        The electron loss function we intend to use. The form of the parameters
        should be: elf(k, omega, *elfparams), where k is the momentum variable
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

    Returns
    -------
    L : float
        The stopping number for a given initial velocity, v.
    '''
    
    # Energy (inner) integral part
    def omegaint(k, v):   
        omegaintegrand = lambda x, y : x * elf(k, x, *elfargs)
        if k*v < wmin:
            return 0
        I = solve_ivp(omegaintegrand, (wmin, k*v), [0], rtol=wtol[0], 
                      atol=wtol[1])
        return I.y[0][-1]

    # Momentum (outer) integral part
    kintegrand = lambda k, y : 1/k * omegaint(k, v)
    I = solve_ivp(kintegrand, (kmin, 2*v), [0])
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