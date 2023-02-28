# Standard imports
import numpy as np

def log_derivative(f:callable, x:float, dx=1e-3) -> float:
    '''
    Calculate the logarithmic derivative of f(x) at the point x: dln(f)/dln(x)
    Here f(x) is a callable function and x is the value at which to evaluate the derivative
    '''
    dlnf = np.log(f(x+dx/2.)/f(x-dx/2.)) # Two-sided difference in numerator
    dlnx = np.log((x+dx/2.)/(x-dx/2.))   # Two-sided (is this necessary?)
    #dlnx = log(1.+dx/x) # Using this is probably fine; for dx<<x they are equal
    return dlnf/dlnx


def derivative_from_samples(x:float, xs:np.array, fs:np.array) -> float:
    '''
    Calculates the derivative of the function f(x) which is sampled as fs at values xs
    Approximates the function as quadratic using the samples and Legendre polynomials
    TODO: Can this be made to work for array x argument? Vectorize?
    Args:
        x: Point at which to calculate the derivative
        xs: Sample locations
        fs: Value of function at sample locations
    '''
    from scipy.interpolate import lagrange
    ix, _ = find_closest_index_value(x, xs)
    if ix == 0:
        imin, imax = (0, 1) if x < xs[0] else (0, 2) # Tuple braces seem to be necessarry here
    elif ix == len(xs)-1:
        nx = len(xs)
        imin, imax = (nx-2, nx-1) if x > xs[-1] else (nx-3, nx-1) # Tuple braces seem to be necessarry here
    else:
        imin, imax = ix-1, ix+1
    poly = lagrange(xs[imin:imax+1], fs[imin:imax+1])
    return poly.deriv()(x)
#derivative_from_samples = np.vectorize(derivative_from_samples, excluded=['xs', 'fs']) # TODO: Does not work, not sure why.


def logspace(xmin:float, xmax:float, nx:int) -> np.ndarray:
    '''
    Return a logarithmically spaced range of numbers
    '''
    return np.logspace(np.log10(xmin), np.log10(xmax), nx)


def trapz2d(F:np.array, x:np.array, y:np.array) -> float:
    '''
    Two-dimensional trapezium rule
    First integrates along x for each y, and then y
    '''
    Fmid = np.trapz(F, x)
    return np.trapz(Fmid, y)


def find_closest_index_value(x:float, xs:np.array) -> tuple:
    '''
    Find the index, value pair of the closest values in array 'arr' to value 'x'
    '''
    idx = (np.abs(xs-x)).argmin()
    return idx, xs[idx]


def is_array_monotonic(x:np.array) -> bool:
    '''
    Returns True iff the array contains monotonically increasing values
    '''
    return np.all(np.diff(x) > 0.)

