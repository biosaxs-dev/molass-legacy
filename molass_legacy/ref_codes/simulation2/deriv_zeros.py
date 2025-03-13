import sys
import os
import numpy    as np
import matplotlib.pyplot    as plt
from scipy.optimize import fsolve

def f_numerator( xR ):
    return np.sin( xR ) - xR*np.cos( xR )

if len(sys.argv) < 3:
    R = 23
    z = 0.2
else:
    R = float(sys.argv[1])
    z = float(sys.argv[2])

print( 'R=', R, 'z=', z )

zeros_init = np.array( [z] )
x0R = fsolve(f_numerator, zeros_init*R )
x0 = x0R/R
# print( 'x0R=', x0R, x0, f_numerator(x0R) )

print( 'x0=', x0 )

x = np.linspace( 0.005, 0.6, 800 )

plt.plot( x, f_numerator( x*R ) )
plt.plot( x0, np.zeros( len(x0) ), 'or' )
plt.show()
