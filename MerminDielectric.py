#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:22:36 2019

@author: tommy

Numerically calculates the Mermin dielectric function.


This code follows the work by David Perkins, Andre Souza, Didier Saumon, 
and Charles Starrett as produced in the 2013 Final Reports from the Los 
Alamos National Laboratory Computational Physics Student Summer Workshop, 
in the section titled "Modeling X-Ray Thomson Scattering Spectra 
of Warm Dense Matter".
"""


import numpy as np

def realintegrand(p, k, omega, nu, kBT, mu):
    """
    The integrand present in the formula for the real part of the general
    RPA dielectric function.
    
    Parameters:
    ___________
    p: scalar
        The integration variable, which is also the momemtum of the electronic
        state.
    k: scalar
        The change of momentum for an incident photon with momentum k0 
        scattering to a state with momentum k1: k = |k1-k0|, in a.u.
    omega: scalar
        The change of energy for an incident photon with energy w0 
        scattering to a state with energy w1: w = w0-w1, in a.u.
    kBT: scalar
        Thermal energy (kb - Boltzmann's constant, T is temperature) in a.u.
    mu: scalar
        Chemical potential in a.u.    
    nu: scalar
        Collision frequency in a.u.         
    Returns:
    ________
    """
    
    # delta will help with avoiding singularities if the real part of nu is 0.
    deltamod = (1e-11)**(1/2.)
    
    
    # variables to avoid verbose lines later on.
    pp = (k**2 + 2*(omega-nu.imag) + 2*p*k)**2 + (2*nu.real + deltamod)**2
    pm = (k**2 + 2*(omega-nu.imag) - 2*p*k)**2 + (2*nu.real + deltamod)**2
    mp = (k**2 - 2*(omega-nu.imag) + 2*p*k)**2 + (2*nu.real + deltamod)**2
    mm = (k**2 - 2*(omega-nu.imag) - 2*p*k)**2 + (2*nu.real + deltamod)**2
    
    logpart = np.log(np.sqrt(pp/pm)) + np.log(np.sqrt(mp/mm))
    
    FD = 1/(1+np.exp((p**2/2 - mu)/kBT))
    
    return logpart * FD * p

def DEtransform(u, k, omega, nu, kBT, mu, plim):
    """
    Transform the real integral using a Double Exponential (DE) change of
    variables.
    
    The transformation is of the form
    
    x = tanh(a sinh(u))
    dx = a cosh(u) / cos^2(a sinh(u)) du
    
    where a = \pi/2.
    
    This transformation will take an integral from [-1, 1] to
    (-\infty, +\infty).
    
    We first use the linear tranformation 
    
    2p = (b-a)x + (b+a),
    
    where plim=(a,b) are the limits of the original integral, to transform the
    integration range to (-1, 1).
    
    Parameters:
    ___________
    plim: list-like of length 2
        Original limits of integration. The limits after the transformation
        will be from (-\infty, +\infty)
    """
    
    a, b = plim 
    
    ptrans = ((b-a) * np.tanh(np.pi/2*np.sinh(u)) + (b+a))/2
    
    transfactor = (b-a)/2 * np.pi/2 * np.cosh(u)/np.cosh(np.pi/2*np.sinh(u))**2
    
    return transfactor * realintegrand(ptrans, k, omega, nu, kBT, mu)
    

def imagintegrand(p, k, omega, nu, kBT, mu):
    """
    The integrand present in the formula for the imaginary part of the general
    RPA dielectric function.
    
    Parameters:
    ___________
    p: scalar
        The integration variable, which is also the momemtum of the electronic
        state.
    k: scalar
        The change of momentum for an incident photon with momentum k0 
        scattering to a state with momentum k1: k = |k1-k0|, in a.u.
    omega: scalar
        The change of energy for an incident photon with energy w0 
        scattering to a state with energy w1: w = w0-w1, in a.u.
    kBT: scalar
        Thermal energy (kb - Boltzmann's constant, T is temperature) in a.u.
    mu: scalar
        Chemical potential in a.u.   
    nu: scalar
        Collision frequency in a.u. 
    
    Returns:
    ________
    """
    
    # variables to avoid verbose lines later on.
    pp = k**2 + 2*(omega-nu.imag) + 2*p*k
    pm = k**2 + 2*(omega-nu.imag) - 2*p*k
    mp = k**2 - 2*(omega-nu.imag) + 2*p*k
    mm = k**2 - 2*(omega-nu.imag) - 2*p*k
    
    arctanpart = np.arctan2(2.*nu.real, pp) - np.arctan2(2.*nu.real, pm) \
               + np.arctan2(-2.*nu.real, mp) - np.arctan2(-2.*nu.real, mm)
        
    FD = 1/(1+np.exp((p**2/2 - mu)/kBT))
    
    return arctanpart * FD * p  
    #return  arctanpart

def generalRPAdielectric(k, omega, nu, kBT, mu):
    """
    Numerically calculates the dielectric function  in Random Phase 
    Approximation (RPA), epsilon_{RPA}(k, omega + i*nu). This function is 
    labelled general becuase the frequency argument is made complex to account
    for collisions due to ions. This alone is not a correct expression for the
    dielectric function, and is used in calculating the Mermin dielectric 
    function.
    
    Parameters:
    ___________
    k: scalar
        The change of momentum for an incident photon with momentum k0 
        scattering to a state with momentum k1: k = |k1-k0|, in a.u.
    omega: scalar or array-like
        The change of energy for an incident photon with energy w0 
        scattering to a state with energy w1: w = w0-w1, in a.u.
    nu: scalar or array-like
        Collision frequency in a.u. If array-like, must has same size as omega. 
    kBT: scalar
        Thermal energy (kb - Boltzmann's constant, T is temperature) in a.u.
    mu: scalar
        Chemical potential in a.u.
        
    Returns:
    ________
    """
    
    # To handle both scalar and array inputs
    k = np.asarray(k)
    omega = np.asarray(omega)
    nu = np.asarray(nu)
    scalar_input = False
    if k.ndim == 0:
        k = np.expand_dims(k, axis=0) # Makes k 1D
        scalar_input = True
    if omega.ndim == 0:
        omega = np.expand_dims(omega, axis=0)
        scalar_input = True
    if nu.ndim == 0:
        nu = np.expand_dims(nu, axis=0)
        scalar_input = True
    # Length of omega
    N = len(omega)
        
    # A small nu causes some problems when integrating the real and imaginary 
    # parts of the dielectric. 
    # When nu is small, the imaginary integrand is like a modulated step 
    # function between p1 and p2, while the real part develops sharp peaks at
    # p1 and p2 (the peaks should go to infinity, but I damp them with the 
    # small delta term in the integrand).
    p1 = abs(k**2-2*omega)/(2*k)
    p2 = (k**2 + 2*omega)/(2*k)
    p3 = np.sqrt(abs(2*mu))
    
    ### Integral for real part of the dielectric function ###
    
    # # Transformed integrand for real part
    realint = lambda x, lims : DEtransform(x, k, omega, nu, kBT, mu, lims)
    # realint = lambda x : realintegrand(x, k, omega, nu, kBT, mu)
    # All transformed integrations fall roughly within the same region in the
    # transformed space
    t = np.linspace(-2.5*np.ones(N), 2.5*np.ones(N), 200)
    tempwidth = np.sqrt(2*np.abs(mu + 10*kBT))
    realsolve =   np.trapz(realint(t, (np.zeros(N) , p1)), t, axis=0) \
                + np.trapz(realint(t, (p1, p2)), t, axis=0) \
                + np.trapz(realint(t, (p2, 2*p2 + tempwidth)), t, axis=0)


    ### Integral for the imag part of the dielectric function ###
    
    # Integration regions, taking into account difficult points
    # (2*p2+10 should effectively act like infinity).
    width = nu.real + 1e-11
    # n = 200
    # reg1 = np.linspace(np.zeros(N), p1 - width, n)
    # reg2 = np.linspace(p1 - width , p1 + width, n//2)
    # reg3 = np.linspace(p1 + width , p2 - width, n)
    # reg4 = np.linspace(p2 - width , p2 + width, n//2)
    # reg5 = np.linspace(p2 + width , p2 + width + tempwidth, n) 
    
    imagint = lambda p : imagintegrand(p, k, omega, nu, kBT, mu)
    
    # imagsolve =   np.trapz(imagint(reg1), reg1, axis=0) \
    #             + np.trapz(imagint(reg2), reg2, axis=0) \
    #             + np.trapz(imagint(reg3), reg3, axis=0) \
    #             + np.trapz(imagint(reg4), reg4, axis=0) \
    #             + np.trapz(imagint(reg5), reg5, axis=0)

    # Explicitly identify difficult points in integration range, plus the
    # "widths" around each point
    nuwidth = nu.real + 1e-11
    # Put these into an array
    pdiff = np.zeros((8,N))
    pdiff[1] = np.maximum(p1 - nuwidth, np.zeros(N))
    pdiff[2] = p1 + nuwidth
    pdiff[3] = np.maximum(p2 - nuwidth, np.zeros(N))
    pdiff[4] = p2 + nuwidth
    pdiff[5] = np.maximum(p3 - tempwidth, np.zeros(N))
    pdiff[6] = p3 + tempwidth
    pdiff[7] = p2 + nuwidth + tempwidth
    
    # Sort the difficult points, so they are in order
    pdiff = np.sort(pdiff, axis=0)
    # Linearly interpolate between the difficult point +/- their widths to
    # create a set of integration regions bounded by these points +/- widths
    intregions = np.linspace(pdiff[0:7], pdiff[1:8], 25)
    # Integrate within each of the regions
    imagintegrateregions = np.trapz(imagint(intregions), intregions, axis=0)
    # Add up the integrations between each region, resulting in an array of
    # length N
    imagsolve = np.sum(imagintegrateregions, axis=0)
    
    ret = 1j * 2 / np.pi / k**3 * imagsolve
    ret += 1 + 2 / np.pi / k**3 * realsolve

    if scalar_input:
        return np.squeeze(ret)
    return ret

def generalMermin(epsilon, k, omega, nu, *args):
    """
    Numerically calculates the Mermin dielectric function. This adds some ionic
    structure to the dielectric function passed through epsilon. Typically this
    will be the RPA dielectric function, but we also want to allow for a 
    general dielectric functions.
    
    Parameters:
    ___________
    epsilon: function
        dielectric function that we want to add ionic information to. The 
        argument structure must be epsilon(k, omega, nu, args) and args must
        be ordered properly.
    k: scalar
        The change of momentum for an incident photon with momentum k0 
        scattering to a state with momentum k1: k = |k1-k0|, in a.u.
    omega: scalar
        The change of energy for an incident photon with energy w0 
        scattering to a state with energy w1: w = w0-w1, in a.u.
    nu: scalar
        Collision frequency in a.u. 
    args: tuple
        Additional arguments (temperature, chemical potential, ...). Must be 
        same order as in the epsilon() function.
    """
    
    epsnonzerofreq = epsilon(k, omega, nu, *args)
    epszerofreq    = epsilon(k, 0, 0, *args)
    
    # If both nu is zero, expect epsnonzerofreq. But if omega also equals zero,
    # this code fails. Add a little delta to omega to avoid this.
    delta = 1e-10
    numerator   = ((omega + delta) + 1j*nu)*(epsnonzerofreq - 1)
    denominator = (omega+delta) \
                  + 1j*nu * (epsnonzerofreq - 1)/(epszerofreq - 1)
    
    
    return 1 + numerator/denominator
    
def MerminDielectric(k, omega, nu, kBT, mu):
    """
    Numerically calculates the Mermin dielectric, which builds upon the RPA
    dielectric function by taking into account electron collisions with ions.
    
    Parameters:
    ___________
    k: scalar
        The change of momentum for an incident photon with momentum k0 
        scattering to a state with momentum k1: k = |k1-k0|, in a.u.
    omega: scalar
        The change of energy for an incident photon with energy w0 
        scattering to a state with energy w1: w = w0-w1, in a.u.
    kBT: scalar
        Thermal energy (kb - Boltzmann's constant, T is temperature) in a.u.
    mu: scalar
        Chemical potential in a.u.
    nu: scalar
        Collision frequency in a.u. 
        
    Returns:
    ________
    """
    
    return generalMermin(generalRPAdielectric, k, omega, nu, kBT, mu)

def ELF(k, omega, nu, kBT, mu):
    """
    Electron Loss Function, related to the amount of energy dissapated in the 
    system.
    """
    
    eps = MerminDielectric(k, omega, nu, kBT, mu)
    return eps.imag/(eps.real**2 + eps.imag**2)

# Tests
if __name__=='__main__':
    import matplotlib.pyplot as plt    

    # k = 0.05
    # T = 0.3
    # mu = 0.3
    # nu = 0.0
    # w = np.linspace(0, 0.6, 500)
    # eps = generalRPAdielectric(k, w, nu, T, mu)
    # #plt.plot(w, eps.imag)
    # # plt.plot(w, eps.real)
    # plt.plot(w, eps.imag / (eps.imag**2 + eps.real**2))
    # p1 = abs(k**2-2*w)/(2*k)
    # p2 = (k**2 + 2*w)/(2*k)
    # p3 = np.sqrt(2*np.abs(mu))
    # tp3 = (2*p3 - (p2+p1))/(p2-p1)
    # print(tp3)
    # plt.plot(t, DEtransform(t, k, w, nu, T, mu , (p2, 2*p2+10)))



    
