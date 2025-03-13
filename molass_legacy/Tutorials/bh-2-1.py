# basin hopping global optimization for the ackley multimodal objective function
from scipy.optimize import basinhopping
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
from matplotlib import animation

# objective function
def objective(v):
    x, y = v
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# define range for input
r_min, r_max = -5.0, 5.0
# define the starting point as a random sample from the domain

minima_list = []
init_points = [(-4, -4), (-4, -4), (-3, 4), (-3, 4), (4, -3), (4, -3)]

def solve(k):
    # pt = r_min + rand(2) * (r_max - r_min)
    pt = init_points[k]
    fv = objective(pt)
    minima_list.append((pt, fv, True))

    if True:
        def minimum_callback(x, fv, accept):
            # print(x, fv, accept)
            minima_list.append((x, fv, accept))

        # perform the basin hopping search
        result = basinhopping(objective, pt, stepsize=0.5, niter=18, callback=minimum_callback)
    else:
        from scipy.optimize import differential_evolution
        def minimum_callback(xk, convergence=0):
            # print(x, fv, accept)
            fv = objective(xk)
            minima_list.append((xk, fv, convergence))

        # define range for input
        r_min, r_max = -5.0, 5.0
        # define the bounds on the search
        bounds = [[r_min, r_max], [r_min, r_max]]
        # perform the differential evolution search
        result = differential_evolution(objective, bounds, callback=minimum_callback)
    return result

for k in range(len(init_points)):
    result = solve(k)
    # summarize the result
    print('Status : %s' % result['message'])
    print('Total Evaluations: %d' % result['nfev'])
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

fig.suptitle("Demonstration of Basin Hopping Algorithm", fontsize=20)
ax.set_title("Surface Plot", fontsize=16)
ax2.set_title("Contour Plot", fontsize=16)
fig.tight_layout()

surface = ax.plot_surface(xx, yy, results, cmap='jet', alpha=0.3)
contuor = ax2.contourf(xx, yy, results, cmap='jet', alpha=0.3)
# show the plot

minima_array = np.array([np.concatenate([x, [v]]) for x, v, a in minima_list])
x, y, z = minima_array[0]
points, = ax.plot([x], [y], [z], 'o', color='cyan')
trace3, = ax.plot(*minima_array[0:1,:].T, color='red')
point, = ax2.plot(x, y, 'o', color='cyan')
trace2, = ax2.plot(*minima_array[0:1,0:2].T, color='red')

print("len(minima_list)=", len(minima_list))

def reset():
    x, y, z = minima_array[0]
    points.set_data(np.array([x]), np.array([y]))
    points.set_3d_properties(np.array([z]), 'z')
    trace3.set_data(*minima_array[0:1,0:2].T)
    trace3.set_3d_properties(minima_array[0:1,2], 'z')
    point.set_data([x], [y])
    trace2.set_data(*minima_array[0:1, 0:2].T)
    return points, trace3, point, trace2

def animate(i):
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

anim = animation.FuncAnimation(fig, animate, frames=len(minima_list), blit=True, init_func=reset, interval=200)
# anim.save("anim.gif")

plt.show()
