import numpy                as np
import matplotlib.pyplot    as plt
import statsmodels.api      as sm

def draw( ax, N, A, B, E ):
    x = np.linspace( 0, 1, N )
    y0  = A + B*x
    y_  = y0 + E * np.random.randn( N )

    stdev = np.std( y_ )
    print( 'std(y_)=', stdev )

    X = sm.add_constant( x )
    result = sm.OLS( y_, X ).fit()
    print( 'result.params=', result.params )
    Va, Vb = np.sqrt( np.diag( result.cov_params() ) )
    print( 'sqrt(cov_params)=', Va, Vb )
    # print( 'result.resid=', result.resid )
    ave_resid = np.sqrt( np.average( result.resid**2 ) )
    print(  'sqrt( average( resid**2 ) )=', ave_resid )


    A_, B_ = result.params
    yr  = A_ + B_ * x

    ax.set_title( 'y=a+b*x; n=%d, a=%.1g, b=%.1g, E=%.1g' % ( N, A, B, E ) )
    ax.set_ylim( 0, 12 )
    if B < 5:
         tx, ty = 0.05, 7
    else:
         tx, ty = 0.5, 1
    ax.text( tx, ty,
                'STDEV=%.3g\nAveraged Error=%.3g\nVar(a_)=%.3g\nVar(b_)=%.3g' %
                ( stdev, ave_resid, Va, Vb )
           )
    ax.plot( x, y_, 'b'  )
    ax.plot( x, y0, 'y' )
    ax.plot( x, yr, 'r' )

fig = plt.figure( figsize=( 18, 9 ) )

axes = [ fig.add_subplot( 4, 4, i+1 ) for i in range(16) ]

A   = 5
E   = 0.2

i = 0
for N in [ 10, 100, 1000, 10000 ]:
    for B in [ 0, 3, -3, 7 ]:
        draw( axes[i], N, A, B, E  )
        i += 1

plt.tight_layout()
plt.show()
