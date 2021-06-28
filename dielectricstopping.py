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
from scipy.integrate import quad_vec
import scipy.optimize as opt

def plasmafreq(density):
    return np.sqrt(4*np.pi*density)

def sumrule(density):
    return np.pi/2 * plasmafreq(density)**2

def genELF(dielfunc, k, w):
    eps = dielfunc(k, w)
    return eps.imag / (eps.imag**2 + eps.real**2)

def modBG_wp(den, k, temp):
    """
    Modified Bohm-Gross dispersion relation, given in Glezner & Redmer, Rev.
    Mod. Phys., 2009, Eqn (16).
    
    den - electron density (au)
    k - wavenumber (au)
    temp - thermal energy (au)
    """  
    wp    = np.sqrt(4*np.pi*den)
    BG_wp = np.sqrt(wp**2 + 3*temp*k**2)
    thermal_deBroglie = np.sqrt(2*np.pi/temp)
    
    return np.sqrt(BG_wp**2 + 3*temp*k**2 \
                    * 0.088 * den * thermal_deBroglie**3 \
                    + (k**2/2)**2)

def ELFmax(dielfunc, k, prevroot, prevfun, directopt=True):
    """
    Finds the maximum position of the electron loss function (ELF).
    
    Note: this will stop working for small values of k, so be wary!
    One way to make sure that this function is still working is to check that
    the ELF maximum for a larger k is less than the ELF maximum for a smaller
    k (for the RPA dielectric function).
    
    """
    f = lambda x: -genELF(dielfunc, k, x)
    
    root = 0
    feval = 0
    bounds = (0, prevroot)
    if directopt:
        # Look for minimum of ELF by optimizing the ELF directly
        boundsroot = opt.minimize_scalar(f, bounds=bounds, method='bounded')
        # if -boundsroot.fun <= prevfun:
        #     directopt = False
        # else:
        #     root = boundsroot.x
        #     feval = -boundsroot.fun
        root = boundsroot.x
        feval = -boundsroot.fun
    if not directopt:
        # Look for the minimum of the ELF by finding the second zero of the
        # real part of the dielectric function.
        reeps = lambda x : dielfunc(k, x).real
        # Find the zero
        root = opt.newton(reeps, prevroot)
        feval = -f(root)

    return root, feval, directopt

def omegaintegral(dielfunc, v, k, collfreq, temp, chempot, ELFmaxpos,
                  ELFmaxval, density, vlow=0):
    """
    Calculates the inner, omega integral for the dielectric stopping power

    Parameters
    ----------
    dielfunc : function, of the form f(x, y)
        Dielectric function.
    v : scalar
        Initial charged particle velocity.
    k : scalar
        Wavenumber.
    temp : scalar
        temperature.
    chempot : scalar
        Chemical potential.
    ELFmaxpos: scalar
        Position in omega-space for a given k of the electron loss function
        (ELF).
    ELFmaxval: scalar
        Value of ELF at ELFmaxpos.
    sumrule: scalar
        Sum rule value.
    vlow : scalar, optional
        This parameter is useful for the sequential stopping function. It
        represents the lower integration limit for the omega integral. 
        The default is 0.

    Returns
    -------
    omegaint : float
        Value for the omega integral for a given v, k

    """
    sr = sumrule(density)
    # plasma frequency
    wp = plasmafreq(density)
    
    # A rough lower bound approximate width of peak, most meaningful for small
    # values of k when the ELF is very sharp.
    srwidth = sr / ELFmaxval
       
    # A width associated with the wavenumber, k, and the temperature, temp.
    # This is a conservative upper bound made to capture all of the integrand
    # (with some heuristically motivated approximations).
    kwidth = np.sqrt(2*(10*temp + abs(chempot)))*k + collfreq(0).real*1e3
    
    width = np.minimum(srwidth, kwidth)
      
    # Define our integration regions
    regions = np.zeros(4)
    
    regions[1] =  np.maximum(0., ELFmaxpos - width)
    regions[2] = ELFmaxpos + width
    # This region is important for small k, when srwidth becomes really small.
    regions[3] = np.maximum(ELFmaxpos + kwidth, k*v)
    
    
    
    # Where to place k*v with respect to the regions above
    kvi = np.searchsorted(regions, k*v)
    regions = np.insert(regions, kvi, k*v)

    # integrand
    f = lambda x, y : x * genELF(dielfunc, k, x)
    
    # integral from [0, kv] and from [0, \infty)
    # If kvi == 0, then k*v = 0, don't need to do this integral
    omegaint = 0.
    omegaint_allspace = 0.
    
    for i in range(1, len(regions)):
        I = solve_ivp(f, (regions[i-1], regions[i]), [0],
                      rtol=1e-6, vectorized=True)
        # print("nevals{} = {}".format(i, I.nfev))
        omegaint_allspace += I.y[0][-1]
        # w = np.linspace(regions[i-1], regions[i], 500)
        # omegaint_allspace += np.trapz(f(w), w)
        
        if kvi == i:
            omegaint = omegaint_allspace
        

    # Check the sum rule
    error = (sr - omegaint_allspace)/sr
        
    
    return omegaint, width, error, regions
        

def omegaintegral_check(dielfunc, v, collfreq, temp, chempot, density,
                        kgrid=None):
    """
    Function that makes it easy to check the omega integral by eye (not cool!)

    Parameters
    ----------
    v : TYPE
        DESCRIPTION.
    dielfunc : TYPE
        DESCRIPTION.
    temp : TYPE
        DESCRIPTION.
    chempot : TYPE
        DESCRIPTION.
    density : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    sr = sumrule(density)
    wp = plasmafreq(density)
    
    # String if we don't satisfy the sum rule
    errmssg = ""
    
    ## Upper limit for k integral
    # tempwidth is a temperature-based correction to the typically upper bound
    # seen in the literature.
    tempwidth = np.sqrt(2*(10*temp + abs(chempot)))                                 
    kupperbound= 2*(v +  tempwidth)
    # Define grid in k-space
    if kgrid is None:
        kgrid = np.geomspace(5e-2, kupperbound, 100)
    
    omegaint = np.zeros(len(kgrid))
    error    = np.zeros(len(kgrid))
    directopt = True
    
    # inital guess
    ELFmaxpos = modBG_wp(density, kgrid[-1], temp)
    ELFmaxval = -1
    
    # Find maximum positions of ELF, working backwards starting from larger
    # values of k.
    for i, k  in reversed(list(enumerate(kgrid))):
        prevpos =  ELFmaxpos + tempwidth * kgrid[-1]
        ELFmaxpos, ELFmaxval, directopt = ELFmax(dielfunc, k, prevpos,
                                                 ELFmaxval, directopt)
        # if prevval > ELFmaxval:
        #     break

        omegaint[i], delta, error[i], reg = omegaintegral(dielfunc, v, k, 
                                                          collfreq, temp, 
                                                          chempot, ELFmaxpos,
                                                          ELFmaxval, density)
        
        SRsatisfied = abs(error[i]) < 5e-2
        
        if (not SRsatisfied):
            omegaint[i] = -omegaint[i]
            errmssg = "########## SUMRULE NOT SATISFIED ############\n"\
                    + "k = {:.15f}\n".format(k)\
                    + "ELF max pos = {}\n".format( ELFmaxpos)\
                    +"regions = {}\n".format( reg)\
                    + "ELF max val = {}\n".format( ELFmaxval)\
                    + "error = {:.3f}\n".format(error)
            print(errmssg)

        # delta is roughly related to how sharp/thin the ELF peak is.
        # When it is small *enough*, we treat it as a delta function centered
        # at the plasma frequency, wp. The integrals can then be approximately
        # done analytically.
        # if (not SRsatisfied) and delta < 1e-5:
        #     omegaint[(kgrid*v >= wp - delta)*(kgrid < k)] = sr
        #     break
                
    
    return omegaint, kgrid, error
    # kintegrand = 1/kgrid * omegaint
    # kintegral = np.trapz(kintegrand, kgrid)

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



if __name__=='__main__':
    import fdint
    import dielectricfunction_symln.Mermin.MerminDielectric as MD
    import matplotlib.pyplot as plt
    
    t = 0.03
    mu = 0.3
    dielfunc = lambda k, w : MD.MerminDielectric(k, w, 0.1, t, mu)
    
    den = (2*t)**(3/2) / (2 * np.pi**2) * fdint.fdk(k=1/2., phi=mu/t)
    
    I, k = omegaintegral_check(dielfunc, 6, t, mu, den)
    plt.plot(k, I)
