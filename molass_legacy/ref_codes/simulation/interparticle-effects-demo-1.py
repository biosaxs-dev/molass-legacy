import numpy                as np
import matplotlib.pyplot    as plt
from Intensity              import Intensity

x = np.arange( 0, 0.8, 1e-3 )

# intensity = Intensity( 0.05, 26 )
intensity = Intensity( 0.2, 26 )

y1 = intensity.term1( x )
y2 = intensity.term2( x )

ii = y1 * y2

Y1 = np.log( y1 )
Y_ = np.log( y1 - ii )

X = x**2

fig = plt.figure( figsize=(16,6) )
ax1 = fig.add_subplot( 121 )
ax2 = fig.add_subplot( 122 )

ax1.set_title( 'Intensity plot of a simple model' )
ax2.set_title( 'Intensity plot of a simple model zoomed in a small-q region' )

for ax in [ax1, ax2]:
    ax.plot( X, Y_, color='red' )
    ax.plot( X, Y1, ':', color='blue' )
    ax.set_xlabel( 'Q' )
    ax.set_ylabel( 'Intensity' )

xmin1, xmax1 = ax1.get_xlim()
ymin1, ymax1 = ax1.get_ylim()

xmin2 = xmin1
xmax2 = xmax1/32

ax2.set_xlim( xmin2, xmax2 )
zoom_size = x.shape[0] // 32

# YP = np.hstack( [ Y1[y1 > 0][0::zoom_size],  Y_[y1 - ii > 0][0:zoom_size] ] )
YP = Y1[y1 > 0][0::zoom_size]

yminp = np.min( YP )
ymaxp = np.max( YP )
ymin2 = (yminp + ymaxp)/2
ymax2 = ymaxp
ax2.set_ylim( ymin2, ymax2 )

ax1.plot(   [ xmin2, xmax2, xmax2, xmin2, xmin2 ],
            [ ymin2, ymin2, ymax2, ymax2, ymin2 ], ':', color='black' )

plt.tight_layout()
plt.show()
