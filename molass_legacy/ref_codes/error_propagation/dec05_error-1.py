import sys
import os
import numpy    as np
from mpl_toolkits.mplot3d   import Axes3D
import matplotlib.pyplot    as plt
# import statsmodels.api      as sm

sys.path.append( os.path.dirname( os.path.abspath( __file__ ) ) + '/../../lib' )
import molass_legacy.KekLib, AutorgKek, SerialAnalyzer
import OurStatsModels       as sm
from SerialData             import SerialData
from molass_legacy.KekLib.NumpyUtils             import np_savetxt

sys.path.append( os.path.dirname( os.path.abspath( __file__ ) ) + '/../../..' )
from TestEnv import env_dict

data_folder = env_dict[ 'Dec05' ]
conc_folder = env_dict[ 'Dec05' ]

serial = SerialData( conc_folder, data_folder,  )
serial.wait_until_ready()
serial.make_adjusted_array( 5.0 )

print( 'serial.intensity_array.shape=', serial.intensity_array.shape )
print( 'serial.conc_array.shape=', serial.conc_array.shape )

x = serial.qvector
x_slice = slice( 0, 200 )
x_ = x[x_slice]
indeces = np.arange( 206, 233 )

x_i = 50
x_f = x_[x_i]

fig = plt.figure( figsize=(18, 9) )
ax1 = fig.add_subplot( 231 )
ax2 = fig.add_subplot( 232, projection='3d' )
ax3 = fig.add_subplot( 233 )
ax4 = fig.add_subplot( 234, projection='3d' )
ax5 = fig.add_subplot( 235 )
ax6 = fig.add_subplot( 236 )

ax1.set_title( 'Concentration Curve' )
ax1.set_xlabel( 'Seq No.' )
ax1.set_ylabel( 'Concentration' )

ax2.set_title( 'Intensity Curves on Q[0:200] and the plane at Q[50]=%.3g' % x_f )
ax2.set_xlabel( 'Q' )
ax2.set_ylabel( 'Seq No.' )
ax2.set_zlabel( 'Intensity' )

ax3.set_title( 'Intensity Variation with errorbar at Q[50]=%.3g' % x_f )
ax3.set_xlabel( 'Seq No.' )
ax3.set_ylabel( 'Concentration' )

ax4.set_title( 'Mutivariate Regression at Q[50]=%.3g' % x_f )
ax4.set_xlabel( 'C' )
ax4.set_ylabel( '-C**2' )
ax4.set_zlabel( 'Intensity' )

ax5.set_title( 'Extrapolated A(q)' )
ax5.set_xlabel( 'Q' )
ax5.set_ylabel( 'Intensity / C' )

ax6.set_title( 'Extrapolated B(q)' )
ax6.set_xlabel( 'Q' )
ax6.set_ylabel( 'Intensity / C**2' )

mc_vector = serial.mc_vector[indeces]
ax1.plot( indeces, mc_vector )

for i in indeces:
    c = serial.mc_vector[i]
    # y_ = np.ones( len(x_) ) * c
    y_ = np.ones( len(x_) ) * i
    z_ = serial.intensity_array[i,x_slice,1]
    ax2.plot( x_, y_, z_ )

ymin, ymax = ax2.get_ylim()
ax2.set_ylim( ymin, ymax )
zmin, zmax = ax2.get_zlim()
ax2.set_zlim( zmin, zmax )
ax2.plot( np.ones(5)*x_f,
            [ymin, ymax, ymax, ymin, ymin ],
            [zmin, zmin, zmax, zmax, zmin ],
            ':', color='black' )
ax2.plot( np.ones(len(mc_vector))*x_f, indeces, serial.intensity_array[indeces,x_i,1], ':', color='red' )

y_f = serial.intensity_array[indeces,x_i,1]
y_e = serial.intensity_array[indeces,x_i,2]
ax3.plot( indeces, y_f, ':', color='red' )
ax3.errorbar( indeces, y_f, yerr=y_e, color='blue', fmt='none' )

x1_ = mc_vector
x2_ = -mc_vector**2
X = np.array( [ x1_, x2_] ).T

param_list = []
error_list = []
for i in range(0,200):
    z_f = serial.intensity_array[indeces,i,1]
    e2  = serial.intensity_array[indeces,i,2]**2
    w   = 1/e2
    model   = sm.WLS(z_f, X, weights=w)
    result  = model.fit()
    param_list.append( result.params )
    # error_list.append( np.diag( result.cov_params() ) )
    error = np.sqrt( np.dot( model.XtWX_inv_XtW**2, e2 ) )
    error_list.append( error )

params = param_list[x_i]

z_f = serial.intensity_array[indeces,x_i,1]
z_r = np.dot( X, params )

ax4.scatter( x1_, x2_, z_f )
ax4.scatter( x1_, x2_, z_r, color='orange' )

xmin, xmax = ax4.get_xlim()
ymin, ymax = ax4.get_ylim()
ax4.set_xlim( xmin, xmax )
ax4.set_ylim( ymin, ymax )

xx, yy = np.meshgrid( np.linspace(xmin, xmax, 10), np.linspace(ymin, ymax, 10) )
zz = xx * params[0] + yy * params[1]
ax4.plot_surface(xx, yy, zz, alpha=0.1 )

params_array = np.array( param_list )
error_array = np.array( error_list )
# np_savetxt( 'params_array.csv', params_array )
np_savetxt( 'error_array.csv', error_array )

A = params_array[:,0]
B = params_array[:,1]
ax5.plot( x_, A, color='red' )
ax6.plot( x_, B, color='red' )
ax5.errorbar( x_f, A[x_i], yerr=error_array[x_i,0], color='blue', fmt='none' )
ax6.errorbar( x_f, B[x_i], yerr=error_array[x_i,1], color='blue', fmt='none' )

for ax in [ax5, ax6]:
    ymin, ymax = ax.get_ylim()
    ax.set_ylim( ymin, ymax )
    ax.plot( [ x_f, x_f ], [ymin, ymax], ':', color='black' )

plt.tight_layout()
plt.show()
