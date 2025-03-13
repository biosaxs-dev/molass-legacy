# basin hopping global optimization for the ackley multimodal objective function
import logging
import numpy as np
from numpy.random import rand
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
from numpy import arange
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import animation
from ultranest import ReactiveNestedSampler 

# objective function
def objective(v):
    x, y = v
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

DIM = 2
# define range for input
r_min, r_max = -5.0, 5.0

lower = np.ones(2) * r_min
upper = np.ones(2) * r_max
def my_prior_transform(cube):
    # transform location parameter: uniform prior
    params = cube * (upper - lower) + lower
    return params

nfev = 0
def my_likelihood(params):
    global nfev
    # print("objective_func_wrapper: par=", par)
    fv = objective(params)
    return -fv

vis_info_list = []
init_points = [(-4, -4), (-4, -4), (4, -4), (4, -4), (4, 4), (4, 4), (-4, 4), (-4, -4)]
# init_points = [(-4, -4)]

def solve(k):
    pt = init_points[k]
    fv = objective(pt)
    vis_info_list.append((pt, fv))

    ncall = 0
    def viz_callback(points, info, region, transformLayer, region_fresh=False):
        nonlocal ncall
        ncall += 1
        if ncall < 20:
            logl = points['logl']
            m = np.argmax(logl)
            x = points['p'][m]
            fv = objective(x)
            vis_info_list.append((x, fv))

    param_names = ["p%d" % i for i in range(DIM)]
    sampler = ReactiveNestedSampler(param_names, my_likelihood, my_prior_transform)
    sampler.logger.setLevel(logging.INFO)       # to suppress debug log    

    result = sampler.run(min_num_live_points=64, max_ncalls=1000, viz_callback=viz_callback)
    opt_params = result['maximum_likelihood']['point']

    # perform the basin hopping search
    return dict(x=opt_params, nfev=nfev, ncall=ncall, message="ok")

history_pos = 0
for k in range(len(init_points)):
    result = solve(k)

    # summarize the result
    print('Status : %s' % result['message'])
    print('Total Evaluations: %d' % result['nfev'])
    print('Callback calls: %d' % result['ncall'])

    # evaluate solution
    solution = result['x']
    evaluation = objective(solution)
    print('Solution: f(%s) = %.5f' % (solution, evaluation))

# sample input range uniformly at 0.1 increments
xaxis = arange(r_min, r_max, 0.1)
yaxis = arange(r_min, r_max, 0.1)
# create a mesh from the axis
xx, yy = meshgrid(xaxis, yaxis)
# compute targets
results = objective((xx, yy)) 
# create a surface plot with the jet color scheme
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

fig.suptitle("Demonstration of Nested Sampling Algorithm using Ackley function", fontsize=20)
ax.set_title("Surface Plot", fontsize=16)
ax2.set_title("Contour Plot", fontsize=16)
fig.tight_layout()

surface = ax.plot_surface(xx, yy, results, cmap='jet', alpha=0.3)
contuor = ax2.contourf(xx, yy, results, cmap='jet', alpha=0.3)
# show the plot

minima_array = np.array([np.concatenate([x, [v]]) for x, v in vis_info_list])
x, y, z = minima_array[0]
points, = ax.plot([x], [y], [z], 'o', color='cyan')
trace3, = ax.plot(*minima_array[0:1,:].T, color='red')
point, = ax2.plot(x, y, 'o', color='cyan')
trace2, = ax2.plot(*minima_array[0:1,0:2].T, color='red', label="local minima trace")
ax2.legend()

def reset():
    x, y, z = minima_array[0]
    points.set_data(np.array([x]), np.array([y]))
    points.set_3d_properties(np.array([z]), 'z')
    trace3.set_data(*minima_array[0:1,0:2].T)
    trace3.set_3d_properties(minima_array[0:1,2], 'z')
    point.set_data([x], [y])
    trace2.set_data(*minima_array[0:1,0:2].T)
    return points, trace3, point, trace2

def animate(i):
    global step_arrow

    x, y, z = minima_array[i]
    points.set_data(np.array([x]), np.array([y]))
    points.set_3d_properties(np.array([z]), 'z')
    point.set_data([x], [y])
    color = 'cyan' if i % 20 == 0 else 'red'
    points.set_color(color)
    point.set_color(color)
    j, r = divmod(i, 20)
    start = j*20
    stop = start + r + 1
    slice_ = slice(start, stop)
    trace3.set_data(*minima_array[slice_,0:2].T)
    trace3.set_3d_properties(minima_array[slice_,2], 'z')
    trace2.set_data(*minima_array[slice_,0:2].T)
    return points, trace3, point, trace2

anim = animation.FuncAnimation(fig, animate, frames=len(vis_info_list), blit=True, init_func=reset, interval=500)
import sys
if len(sys.argv) > 1 and sys.argv[1] == "save":
    anim.save("ns-anim.gif")

plt.show()
