import sys
import os
import numpy                as np
import matplotlib.pyplot    as plt
# import statsmodels.api      as sm
from scipy                  import stats
sys.path.append( os.path.dirname( os.path.abspath( __file__ ) ) + '/../lib' )
import OurStatsModels       as sm


N   = 10
a   = 5
b   = 3
s   = 0.2
x   = np.linspace( 0, 1, N )
np.random.seed( 1234 )
e   = np.random.normal( 0, s, N )
yt  = a * x + b
y   = yt + e

X   = sm.add_constant( x )
model   = sm.WLS( y, X )
result  = model.fit()

print( 'result.params=', result.params )
print( 'result.cov_params=', np.sqrt( np.diag( result.cov_params() ) ) )

"""
# e2 = np.ones( (N,) ) * s**2
e2 =  ( y - np.dot( X, result.params ) )**2
print( 'e2=', e2 )
error = np.sqrt( np.dot( model.XtWX_inv_XtW**2, e2 ) )
print( 'error=', error )
"""

slope, intercept, _, _, std_err = stats.linregress( x, y )
print( 'intercept, slope=', [ intercept, slope ] )
print( 'std_err=', std_err )

param_array = []
for i in range(10000):
    e   = np.random.normal( 0, s, N )
    y   = yt + e
    model   = sm.WLS( y, X )
    result  = model.fit()
    param_array.append( result.params )

param_array_ = np.array( param_array )
print( 'std=', np.std( param_array_, axis=0 ) )
