import numpy                as np
from mpl_toolkits.mplot3d   import Axes3D
import matplotlib.pyplot    as plt
from Intensity              import Intensity

Q = np.arange( 0, 0.8, 1e-3 )
Q_slice = slice( 0, 100 )
C_array = [ 1e-2, 1e-3 ]

X = Q**2

fig = plt.figure( figsize=(24,12) )
ax0 = fig.add_subplot( 231 )
ax1 = fig.add_subplot( 232,  projection='3d' )
ax2 = fig.add_subplot( 233,  projection='3d' )

axes = []
axes.append( [ ax0, ax1, ax2 ] )

ax0 = fig.add_subplot( 234 )
ax1 = fig.add_subplot( 235,  projection='3d' )
ax2 = fig.add_subplot( 236,  projection='3d' )
axes.append( [ ax0, ax1, ax2 ] )

for i, C in enumerate(C_array):
    conc = 0.2 * np.exp( - C * ( np.arange( 100 ) - 50 )**2  )

    Y1_array = []
    Y__array = []

    for c in conc:

        intensity = Intensity( c, 29 )

        y1 = intensity.term1( Q )
        y2 = intensity.term2( Q )

        ii = y1 * y2

        Y1 = np.log( y1 )
        Y_ = np.log( y1 - ii )

        Y1_array.append( Y1 )
        Y__array.append( Y_ )

    ax0, ax1, ax2 = axes[i]
    ax0.set_title( 'Concentration Variation' )
    ax0.set_xlabel( 'Seq-No.' )
    ax0.set_ylabel( 'Concentration' )
    ax1.set_title( 'Guinier plot varying Seq-No.' )
    ax1.set_xlabel( 'Q' )
    ax1.set_ylabel( 'Seq-No.' )
    ax1.set_zlabel( 'Intensity' )
    ax2.set_title( 'Guinier plot varying Concentration' )
    ax2.set_xlabel( 'Q' )
    ax2.set_ylabel( 'Concentration' )
    ax2.set_zlabel( 'Intensity' )

    ax0.plot( conc, color='blue' )

    for i in range( 100 ):
        X_ = X[Q_slice]
        Y1 = Y1_array[i][Q_slice]
        Y_ = Y__array[i][Q_slice]
        I_ = np.ones( ( X_.shape[0], ) ) * i
        C  = np.ones( ( X_.shape[0], ) ) * conc[i]
        ax1.plot( X_, I_, Y_, color='red' )
        ax1.plot( X_, I_, Y1, ':', color='blue' )
        ax2.plot( X_, C, Y_, color='red' )
        ax2.plot( X_, C, Y1, ':', color='blue' )

plt.tight_layout()
plt.show()
