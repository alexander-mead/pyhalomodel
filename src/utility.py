# Standard imports
import numpy as np

### General ###

def log_derivative(f, x, dx=1e-3):
    '''
    Calculate the logarithmic derivative of f(x) at the point x: dln(f)/dln(x)
    '''
    from numpy import log
    dlnf = log(f(x+dx/2.)/f(x-dx/2.)) # Two-sided difference in numerator
    dlnx = log((x+dx/2.)/(x-dx/2.))   # Two-sided (is this necessary?)
    #dlnx = log(1.+dx/x) # Using this is probably fine; for dx<<x they are equal
    return dlnf/dlnx


def logspace(xmin, xmax, nx):
    '''
    Return a logarithmically spaced range of numbers
    '''
    from numpy import logspace, log10
    return logspace(log10(xmin), log10(xmax), nx)


def trapz2d(F, x, y):
    '''
    Two-dimensional trapezium rule
    First integrates along x for each y, and then y
    '''
    from numpy import zeros, trapz
    Fmid = zeros((len(y)))
    for iy, _ in enumerate(y):
        Fmid[iy] = trapz(F[:, iy], x)
    return trapz(Fmid, y)


def findClosestIndex(x, arr):
    '''
    Find the index, value pair of the clostest value in array 'arr' to value 'x'
    '''
    index = (np.abs(arr-x)).argmin()
    return index, arr[index]

### ###

