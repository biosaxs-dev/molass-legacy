# coding: utf-8
"""

    extrapolation-demo-2.py

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF

"""
import sys
import os
import numpy    as np
from mpl_toolkits.mplot3d   import Axes3D
import matplotlib.pyplot    as plt
# import statsmodels.api      as sm
from Intensity              import Intensity

this_dir = os.path.dirname( os.path.abspath( __file__ ) )
sys.path.append( this_dir + '/../../lib' )
import molass_legacy.KekLib
import OurStatsModels       as sm
from molass_legacy.KekLib.NumpyUtils             import np_savetxt

SLICE_NUM_POINTS = 400
LOG_PLOT_A = True

x = np.linspace( 0.005, 0.8, 800 )
C = 1e-2
cvector_all = 0.1 * np.exp( - C * ( np.arange( 50 ) - 25 )**2  )
indeces = np.arange( 0, 25 )

e = np.ones( len(x) ) * 1e-3

intensity_list = []
d_error_list = []
for i in indeces:
    c = cvector_all[i]
    intensity = Intensity( c, 29 )
    ii = intensity( x )
    intensity_list.append( ii )
    d_error_list.append( e )

intensity_array = np.array( intensity_list )
d_error_array = np.array( d_error_list )
print( 'intensity_array.shape=', intensity_array.shape )
print( 'd_error_array.shape=', d_error_array.shape )

x_slice = slice( 0, SLICE_NUM_POINTS )
x_ = x[x_slice]

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
ax4.set_ylabel( 'C²' )
ax4.set_zlabel( 'Intensity' )

ax5.set_title( 'Extrapolated A(q)' )
ax5.set_xlabel( 'Q' )
ylabel_ = 'ln(Intensity / C)' if LOG_PLOT_A else 'Intensity / C'
ax5.set_ylabel( ylabel_ )

ax6.set_title( 'Extrapolated B(q)' )
ax6.set_xlabel( 'Q' )
ax6.set_ylabel( 'Intensity / C²' )


cvector_ = cvector_all[indeces]
ax1.plot( indeces, cvector_ )

for i in indeces:
    c = cvector_all[i]
    # y_ = np.ones( len(x_) ) * c
    y_ = np.ones( len(x_) ) * i
    z_ = intensity_array[i,x_slice]
    ax2.plot( x_, y_, z_ )

ymin, ymax = ax2.get_ylim()
ax2.set_ylim( ymin, ymax )
zmin, zmax = ax2.get_zlim()
ax2.set_zlim( zmin, zmax )
ax2.plot( np.ones(5)*x_f,
            [ymin, ymax, ymax, ymin, ymin ],
            [zmin, zmin, zmax, zmax, zmin ],
            ':', color='black' )
ax2.plot( np.ones(len(cvector_))*x_f, indeces, intensity_array[indeces,x_i], ':', color='red' )

y_f = intensity_array[indeces,x_i]
y_e = d_error_array[indeces,x_i]
ax3.plot( indeces, y_f, ':', color='red' )
# ax3.errorbar( indeces, y_f, yerr=y_e, color='blue', fmt='none' )

x1_ = cvector_
x2_ = cvector_**2
X = np.array( [ x1_, x2_] ).T

param_list = []
error_list = []
for i in range(0,SLICE_NUM_POINTS):
    z_f = intensity_array[indeces,i]
    e2  = d_error_array[indeces,i]**2
    w   = 1/e2
    model   = sm.WLS(z_f, X, weights=w)
    result  = model.fit()
    param_list.append( result.params )
    # error_list.append( np.diag( result.cov_params() ) )
    error = np.sqrt( np.dot( model.XtWX_inv_XtW**2, e2 ) )
    error_list.append( error )

params = param_list[x_i]

z_f = intensity_array[indeces,x_i]
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
# np_savetxt( 'error_array.csv', error_array )

A = params_array[:,0]
B = params_array[:,1]
A_ = np.log(A) if LOG_PLOT_A else A
ax5.plot( x_, A_, color='red', label='Kek A(q)' )
ax6.plot( x_, B, color='red', label='Kek B(q)' )
ax5.errorbar( x_f, A[x_i], yerr=error_array[x_i,0], color='blue', fmt='none' )
ax6.errorbar( x_f, B[x_i], yerr=error_array[x_i,1], color='blue', fmt='none' )

c = 1
intensity = Intensity( c, 29 )
At = intensity.term1( x_ )
Bt = -At*intensity.term2( x_ )

ax5.plot( x_, np.log(At), ':', color='cyan', label='True A(q)' )
ax6.plot( x_, Bt, ':', color='cyan', label='True B(q)' )

# np_savetxt( 'A.csv', A )
# np_savetxt( 'B.csv', B )

for ax in [ax5, ax6]:
    ymin, ymax = ax.get_ylim()
    ax.set_ylim( ymin, ymax )
    ax.plot( [ x_f, x_f ], [ymin, ymax], ':', color='black' )

for ax in [ax5, ax6]:
    ax.legend()

plt.tight_layout()
plt.show()
