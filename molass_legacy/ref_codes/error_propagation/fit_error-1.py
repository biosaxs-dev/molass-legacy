"""
    stack overflow
    Getting standard errors on fitted parameters using the optimize.leastsq method in python
"""
import random
import numpy as np
from scipy import optimize

def f( x, p0, p1, p2):
    return p0*x + 0.4*np.sin(p1*x) + p2

def ff(x, p):
    return f(x, *p)

# These are the true parameters
p0 = 1.0
p1 = 40
p2 = 2.0

# These are initial guesses for fits:
pstart = [
    p0 + random.random(),
    p1 + 5.*random.random(), 
    p2 + random.random()
]

# %matplotlib inline
import matplotlib.pyplot as plt
xvals = np.linspace(0., 1, 120)
yvals = f(xvals, p0, p1, p2)

# Generate data with a bit of randomness
xdata = np.array(xvals)
np.random.seed(42)
err_stdev = 0.2
yvals_err =  np.random.normal(0., err_stdev, len(xdata))
ydata = f(xdata, p0, p1, p2) + yvals_err

plt.plot(xvals, yvals)
plt.plot(xdata, ydata, 'o', mfc='None')
plt.show()

def fit_leastsq(p0, datax, datay, function):

    errfunc = lambda p, x, y: function(x,p) - y

    pfit, pcov, infodict, errmsg, success = \
        optimize.leastsq(errfunc, p0, args=(datax, datay), \
                          full_output=1, epsfcn=0.0001)

    if (len(datay) > len(p0)) and pcov is not None:
        s_sq = (errfunc(pfit, datax, datay)**2).sum()/(len(datay)-len(p0))
        pcov = pcov * s_sq
    else:
        pcov = np.inf

    error = [] 
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_leastsq = pfit
    perr_leastsq = np.array(error) 
    return pfit_leastsq, perr_leastsq 

pfit, perr = fit_leastsq(pstart, xdata, ydata, ff)

print("\nFit paramters and parameter errors from lestsq method :")
print("pfit = ", pfit)
print("perr = ", perr)
