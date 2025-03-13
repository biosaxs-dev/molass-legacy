# coding: utf-8
"""

    ファイル名：    SimulatedAnnealing.py

    処理内容：      Simulated Annealing

    original:
        http://apmonitor.com/me575/index.php/Main/SimulatedAnnealing

    animation, etc. by Masatsuyo Takahashi,

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF
"""
## Generate a contour plot
# Import some other libraries that we'll need
# matplotlib and numpy packages must also be installed
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec    import GridSpec
import matplotlib.animation as animation

import random

##################################################
# Simulated Annealing
##################################################
class SimulatedAnnealing:
    def __init__( self ):
        pass

    def minimize( self, objective_f, xrange=None, start=None, seed=None, xconstaints=None, n=50, m=50 ):

        assert( xrange is not None )
        xrange = np.array( xrange )

        # Start location
        if start is None:
            x_start = np.average( xrange, axis=1 )
        else:
            x_start = start

        # Number of cycles
        # Number of trials per cycle
        # Number of accepted solutions
        na = 0.0
        # Probability of accepting worse solution at the start
        p1 = 0.7
        # Probability of accepting worse solution at the end
        p50 = 0.001
        # Initial temperature
        t1 = -1.0/np.log(p1)
        # Final temperature
        t50 = -1.0/np.log(p50)
        # Fractional reduction every cycle
        frac = (t50/t1)**(1.0/(n-1.0))
        # Initialize x
        x = np.zeros((n+1,2))
        x[0] = x_start
        xi = np.zeros(2)
        # xi = x_start
        na = na + 1.0
        # Current best results so far
        xc = np.zeros(2)
        xc = x[0]
        fc = objective_f(x_start)
        fs = np.zeros(n+1)
        fs[0] = fc

        # Current temperature
        t = t1
        ts = np.zeros(n+1)
        ts[0] = t1

        # DeltaE Average
        DeltaE_avg = 0.0

        if seed is not None:
            random.seed( seed )

        for i in range(n):
            print('Cycle: ' + str(i) + ' with Temperature: ' + str(t))
            for j in range(m):

                while True:
                    # Generate new trial points
                    xi[0] = xc[0] + random.random() - 0.5
                    xi[1] = xc[1] + random.random() - 0.5
                    # Clip to upper and lower bounds
                    xi[0] = max( min(xi[0], xrange[0,1]), xrange[0,0])
                    xi[1] = max( min(xi[1], xrange[1,1]), xrange[1,0])
                    if xconstaints is None or xconstaints( xi ):
                        break

                DeltaE = abs(objective_f(xi)-fc)
                fv = objective_f(xi)
                if np.isfinite(fv):
                    if ( fv > fc ):
                        # Initialize DeltaE_avg if a worse solution was found
                        #   on the first iteration
                        if (i==0 and j==0): DeltaE_avg = DeltaE
                        # objective function is worse
                        # generate probability of acceptance
                        p = np.exp(-DeltaE/(DeltaE_avg * t))
                        # determine whether to accept worse point
                        if (random.random()<p):
                            # accept the worse solution
                            accept = True
                        else:
                            # don't accept the worse solution
                            accept = False
                    else:
                        # objective function is lower, automatically accept
                        accept = True
                else:
                    accept = False
                if accept:
                    # update currently accepted solution
                    xc[0] = xi[0]
                    xc[1] = xi[1]
                    fc = objective_f(xc)
                    # increment number of accepted solutions
                    na = na + 1.0
                    # update DeltaE_avg
                    DeltaE_avg = (DeltaE_avg * (na-1.0) +  DeltaE) / na
            # Record the best x values at the end of every cycle
            x[i+1][0] = xc[0]
            x[i+1][1] = xc[1]
            fs[i+1] = fc
            ts[i+1] = t
            # Lower the temperature for next cycle
            t = frac * t

        self.start  = x_start
        self.n  = n
        self.x  = x
        self.fs = fs
        self.ts = ts
        self.fc = fc
        self.xc = xc


def animate_annealing( field, anneal ):
    x1m, x2m, fm = field

    # print solution
    print('Best solution: ' + str(anneal.xc))
    print('Best objective: ' + str(anneal.fc))

    # Create a contour plot
    fig = plt.figure( figsize=(15, 6) )
    gs = GridSpec( 3, 2 )

    ax0 = fig.add_subplot( gs[:, 0] )
    ax1 = fig.add_subplot( gs[0, 1] )
    ax2 = fig.add_subplot( gs[1, 1] )
    ax3 = fig.add_subplot( gs[2, 1] )

    ax0.set_title('Optimization Field')
    ax1.set_title('Convergence of the Function Value')
    ax2.set_title('Convergence of the Variable Point')
    ax3.set_title('Cooling of the Temperature ')

    # Specify contour lines
    #lines = range(2,52,2)
    # Plot contours
    CS = ax0.contour(x1m, x2m, fm)#,lines)
    # Label contours
    ax0.clabel(CS, inline=1, fontsize=10)
    # Add some text to the plot
    ax0.set_xlabel('x1')
    ax0.set_ylabel('x2')
    ax0.plot( [anneal.start[0], anneal.x[0,0]], [anneal.start[1], anneal.x[0,0]], 'y-o' )
    ax0.plot( anneal.start[0], anneal.start[1], 'ro' )
    line0, = ax0.plot(anneal.x[:,0], anneal.x[:,1], 'y-o')
    # plt.savefig('contour.png')

    line1,  = ax1.plot(anneal.fs,'r.-')
    line21, = ax2.plot(anneal.x[:,0],'b.-')
    line22, = ax2.plot(anneal.x[:,1],'g--')
    line3,  = ax3.plot(anneal.ts,'c.-')

    ax1.legend(['Objective'])
    ax2.legend(['x1','x2'])
    ax3.legend(['Temperature'])

    # Save the figure as a PNG
    # plt.savefig('iterations.png')

    # update the data
    def animate(i):
        line0.set_xdata(anneal.x[0:i,0])
        line0.set_ydata(anneal.x[0:i,1])
        line1.set_xdata(np.arange(i))
        line1.set_ydata(anneal.fs[0:i])
        line21.set_xdata(np.arange(i))
        line21.set_ydata(anneal.x[0:i, 0])
        line22.set_xdata(np.arange(i))
        line22.set_ydata(anneal.x[0:i, 1])
        line3.set_xdata(np.arange(i))
        line3.set_ydata(anneal.ts[0:i])
        return line0, line1, line21, line22, line3

    # Init only required for blitting to give a clean slate.
    def init():
        line0.set_xdata(np.ma.array(anneal.x[:,0],          mask=True))
        line0.set_ydata(np.ma.array(anneal.x[:,1],          mask=True))
        line1.set_xdata(np.ma.array(np.arange(anneal.n+1),  mask=True))
        line1.set_ydata(np.ma.array(anneal.fs,              mask=True))
        line21.set_xdata(np.ma.array(np.arange(anneal.n+1), mask=True))
        line21.set_ydata(np.ma.array(anneal.x[:,0],         mask=True))
        line22.set_xdata(np.ma.array(np.arange(anneal.n+1), mask=True))
        line22.set_ydata(np.ma.array(anneal.x[:,1],         mask=True))
        line3.set_xdata(np.ma.array(np.arange(anneal.n+1),  mask=True))
        line3.set_ydata(np.ma.array(anneal.ts,              mask=True))
        return line0, line1, line21, line22, line3

    ani = animation.FuncAnimation(fig, animate, np.arange(1, anneal.n+1),
                                    init_func=init,
                                    interval=25, blit=True)

    plt.tight_layout()
    plt.show()

def demo():

    # define objective function
    def f(x):
        x1 = x[0]
        x2 = x[1]
        obj = 0.2 + x1**2 + x2**2 - 0.1*np.cos(6.0*3.1415*x1) - 0.1*np.cos(6.0*3.1415*x2)
        return obj

    def xconstaints( x ):
        return x[0] < x[1]

    anneal = SimulatedAnnealing()
    anneal.minimize( f, xrange=[ [-1, 1], [-1, 1] ], start=[0.6, 0.6], seed=1234,
                        xconstaints=None )

    print( 'anneal.start=', anneal.start )

    # Design variables at mesh points
    i1 = np.arange(-1.0, 1.0, 0.01)
    i2 = np.arange(-1.0, 1.0, 0.01)
    x1m, x2m = np.meshgrid(i1, i2)
    fm = np.zeros(x1m.shape)
    for i in range(x1m.shape[0]):
        for j in range(x1m.shape[1]):
            fm[i][j] = 0.2 + x1m[i][j]**2 + x2m[i][j]**2 \
                 - 0.1*np.cos(6.0*3.1415*x1m[i][j]) \
                 - 0.1*np.cos(6.0*3.1415*x2m[i][j])

    animate_annealing( (x1m, x2m, fm), anneal )
