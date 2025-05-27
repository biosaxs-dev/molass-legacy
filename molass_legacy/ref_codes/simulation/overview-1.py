import sys
import os
import numpy    as np
from mpl_toolkits.mplot3d   import Axes3D, proj3d
import matplotlib.pyplot    as plt
# import statsmodels.api      as sm
from Intensity              import Intensity

sys.path.append( os.path.dirname( os.path.abspath( __file__ ) ) + '/../../../lib' )
import molass_legacy.KekLib
import OurStatsModels       as sm
from molass_legacy.KekLib.NumpyUtils             import np_savetxt

max_intensity = 0.03
error_level = 0.002
error_level_100 = error_level / np.sqrt(100)

# np.random.seed( 1234 )

def extrapolate( X, ye_array ):
    a_array = []
    for j in range(len(q)):
        y = ye_array[:,j]
        w = np.ones( (len(y),) ) * 1/error_level_100**2
        model   = sm.WLS(y, X, weights=w)
        result  = model.fit()
        a_array.append( result.params[0] )

    return np.array( a_array )

def plot_extrapolated( ax, y_, color='blue' ):
    ax.set_title( 'extrapolated intensity/c' )
    max_x_intensity = 1.1
    ax.set_ylim( 0, max_x_intensity )
    ax.plot( q, y_, color=color )
    ax.plot( [ q_see, q_see ], [ 0, max_x_intensity ], ':r' )
    ax.plot( q_see, y_[q_see_i], 'or' )

def annotate_error( ax, x, y, text, offset_ratios=[0.1, 0.1], ha='left' ):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xoffset = ( xmax - xmin ) * offset_ratios[0]
    yoffset = ( ymax - ymin ) * offset_ratios[1]

    ax.annotate( text, xy=(x, y),
                    xytext=( x + xoffset, y + yoffset ),
                    ha=ha,
                    arrowprops=dict( headwidth=5, width=0.5, facecolor='black', shrink=0.05),
                    )

fig = plt.figure( figsize=(18, 9) )

axes = (    [ fig.add_subplot( 4, 5, i+1, projection='3d' ) for i in range(4) ] + [ None ]
        +   [ fig.add_subplot( 4, 5, i+6 ) for i in range(5) ]
        +   [ fig.add_subplot( 4, 5, i+11, projection='3d' ) for i in range(4) ] + [ None ]
        +   [ fig.add_subplot( 4, 5, i+16 ) for i in range(5) ]
       )

qsize = 0.3
xmin = -qsize
xmax =  qsize
ymin = -qsize
ymax =  qsize
N = 100
c = 0.01
R = 29

q = np.linspace(0, qsize, N)
q_see_i = 15
q_see = q[q_see_i]
xx, yy = np.meshgrid( np.linspace(xmin, xmax, N), np.linspace(ymin, ymax, N) )
s = np.sqrt( xx**2 + yy**2 )

tt = np.linspace( 0, 2*np.pi, 100 )
tx = q_see * np.cos( tt )
ty = q_see * np.sin( tt )
ts = np.sqrt( tx**2 + ty**2 )

cvector = np.linspace( 0.01, 0.04, 4 )

y0_array = []
ye_array = []

for i, c in enumerate( cvector ):
    intensity = Intensity( c, R )
    zz = intensity( s )
    ax1 = axes[i]
    ax1.set_title( 'model intensity for c=%.2g' % c )
    ax1.set_xlim( -qsize, qsize )
    ax1.set_ylim( -qsize, qsize )
    ax1.set_zlim( 0, max_intensity )
    ax1.plot_surface(xx, yy, zz, alpha=0.1 )
    tz = intensity(ts)
    ax1.plot( tx, ty, tz, color='red' )

    ax2 = axes[i+5]
    ax2.set_title( 'model intensity for c=%.2g' % c )
    ax2.set_xlim( 0, qsize )
    ax2.set_ylim( 0, max_intensity )
    y = intensity( q )
    y0_array.append( y )
    ax2.plot( q, y )
    ax2.plot( [ q_see, q_see ], [ 0, max_intensity ], ':r' )
    ax2.plot( q_see, y[q_see_i], 'or' )

    ax3 = axes[i+10]
    ax3.set_title( 'noisy intensity for c=%.2g' % c )
    ax3.set_xlim( -qsize, qsize )
    ax3.set_ylim( -qsize, qsize )
    ax3.set_zlim( 0, max_intensity )
    ee = error_level * np.random.randn( len(zz) )
    ax3.plot_surface(xx, yy, zz+ee, alpha=0.1 )
    te = error_level * np.random.randn( len(ts) )
    ax3.plot( tx, ty, tz+te, color='red' )

    tx_, ty_, _ = proj3d.proj_transform(tx[0],ty[0],tz[0], ax3.get_proj())
    annotate_error( ax3, tx_, ty_, 'error=%.3g' % ( error_level ),
                        offset_ratios=[0.02, 0.05], ha='center' )

    ax4 = axes[i+15]
    ax4.set_title( 'noisy intensity for c=%.2g' % c )
    ax4.set_xlim( 0, qsize )
    ax4.set_ylim( 0, max_intensity )
    y = intensity( q )
    e = error_level_100 * np.random.randn( len(q) )
    ye = y + e
    ye_array.append( ye )
    ax4.plot( q, y+e )
    ax4.plot( [ q_see, q_see ], [ 0, max_intensity ], ':r' )
    ax4.plot( q_see, ye[q_see_i], 'or' )
    annotate_error( ax4, q_see, ye[q_see_i],
                        r'error=$\frac{%.3g}{\sqrt{100}}$=%.3g' % ( error_level, error_level_100 ),
                        )

X = np.array( [ cvector, -cvector**2 ] ).T
y0_array_ = np.array(y0_array)
ye_array_ = np.array(ye_array)

y0_ = extrapolate( X, y0_array_ )
plot_extrapolated( axes[ 9], y0_ )

ye_ = extrapolate( X, ye_array_ )
plot_extrapolated( axes[19], ye_ )

y_ex_array = []
for i in range(1000):
    ye_array = []
    for c in cvector:
        intensity = Intensity( c, R )
        y = intensity( q )
        e = error_level_100 * np.random.randn( len(q) )
        ye_array.append( y + e )

    y_ = extrapolate( X, np.array(ye_array) )
    y_ex_array.append( y_ )

y_ex_array_ = np.array( y_ex_array )
y_ex_ = np.average( y_ex_array_, axis=0 )
e_ex_ = np.std( y_ex_array_, axis=0 )
ax_ex = axes[19]

ax_ex.errorbar( q, y_ex_, yerr=e_ex_, color='yellow',  fmt='none' )
plot_extrapolated( ax_ex, y_ex_, color='cyan' )
annotate_error( ax_ex, q_see, y_ex_[q_see_i], 'error=?=%.3g' % ( e_ex_[q_see_i] ) )

y = ye_array_[:,q_see_i]
N = len(y)
e2 = np.ones( (N,) ) * error_level_100**2
w = 1/e2

model   = sm.WLS(y, X, weights=w)
result  = model.fit()
param_error = np.sqrt( np.diag( result.cov_params() ) ) 
print( 'param error=', param_error )

# e2 = ( np.dot( X, result.params.T ) - y )**2
print( 'N=', N)
error = np.sqrt( np.dot( model.XtWX_inv_XtW**2, e2 ) )
print( 'error=', error )
print( 'ratio=', param_error/error )


plt.tight_layout()
plt.show()
