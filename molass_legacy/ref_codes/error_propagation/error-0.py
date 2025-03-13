import numpy                as np
import matplotlib.pyplot    as plt
import statsmodels.api      as sm
from scipy.stats            import norm

def regress( N, A, B, E ):
    x = np.linspace( 0, 1, N )
    y0  = A + B*x
    y_  = y0 + E * np.random.randn( N )
    X = sm.add_constant( x )
    result = sm.OLS( y_, X ).fit()
    return result

A   = 5
B   = 3
E   = 0.2

N_regress = 100

fig = plt.figure( figsize=( 12, 9 ) )

varnames = [ 'A', 'B' ]
midvals = [ A, B ]

xlims = []

for n, num_regress in enumerate( [ 10, 100, 1000 ] ):
    ab_error = np.sqrt( np.diag( regress( num_regress, A, B, E ).cov_params() ) )
    print( 'sqrt( cov_params )=', ab_error )
    result_array = [ regress( num_regress, A, B, E ) for i in range(1000) ]
    ab_array = np.array( [ result.params for result in result_array ] )
    ab_error_ = np.average( np.array( [ np.sqrt( np.diag( result.cov_params() ) ) for result in result_array ] ), axis=0 )
    print( 'averaged sqrt( cov_params )=', ab_error_ )

    print( 'simulated stdev=', np.std( ab_array, axis=0 ) )

    for i in range(2):
        ax = fig.add_subplot( 3, 2, n*2+i+1 )
        ax.set_title( 'distribution of estimated %s and normal curve when N=%d' % ( varnames[i], num_regress ) )
        ax.hist( ab_array[:, i], bins=20, normed=True )
        if n == 0:
            xmin, xmax = ax.get_xlim()
            xlims.append( [xmin, xmax] )
        else:
            xmin, xmax = xlims[i]
        ax.set_xlim( xmin, xmax )
        x = np.linspace( xmin, xmax, 100 )
        error = ab_error_[i]
        y = norm.pdf( x, loc=midvals[i], scale=error )
        ax.plot( x, y )
        ymin, ymax = ax.get_ylim()
        ax.text( xmin+0.02*(xmax-xmin), ymax*0.8,
                'error level=%.3g\nerror level / sqrt(%d)=%.3g\nerror=%.3g'
                % ( E, num_regress, E/np.sqrt(num_regress), error ) )

plt.tight_layout()
plt.show()
