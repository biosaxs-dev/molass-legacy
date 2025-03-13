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
from matplotlib.patches import Polygon
from matplotlib import animation

# ---- Take Step Cumtomization Begin -------------------------------------------
from scipy._lib._util import check_random_state
from scipy.optimize._basinhopping import RandomDisplacement, AdaptiveStepsize

takestep_history = []
class StepRecord:
    def __init__(self, x, ret_x, stepsize, adjusted):
        self.x = x
        self.ret_x = ret_x
        self.stepsize = stepsize
        self.adjusted = adjusted

    def get_vertices(self):
        x, y = self.x
        s = self.stepsize
        return [(x-s, y-s), (x+s, y-s), (x+s, y+s), (x-s, y+s)]

    def create_patch(self):
        vertices = self.get_vertices()
        return Polygon(vertices, alpha=0.3, color='green', label="hopping window")

    def move_patch(self, patch):
        vertices = self.get_vertices()
        patch.set_xy(vertices)

    def create_displine(self, ax):
        line, = ax.plot(*np.array([self.x, self.ret_x]).T, color="yellow", label="hopping")
        point, = ax.plot(*self.ret_x, "o", color="orange", label="after hopping")
        return line, point

    def move_displine(self, displace):
        line, point = displace
        xy = np.array([self.x, self.ret_x]).T
        line.set_data(*xy)
        x, y = self.ret_x
        point.set_data([x], [y])

class CustomTakestep(AdaptiveStepsize):
    def __init__(self, stepsize=0.5, interval=50, disp=False, seed=None):
        # set up the np.random generator
        self.rng = check_random_state(seed)
        displace = RandomDisplacement(stepsize=stepsize, random_gen=self.rng)
        AdaptiveStepsize.__init__(self, displace, interval=interval, verbose=disp)

    def take_step(self, x):
        self.nstep += 1
        self.nstep_tot += 1
        adjusted = False
        if self.nstep % self.interval == 0:
            self._adjust_step_size()
            adjusted = True
        x_ = np.copy(x)
        ret_x = self.takestep(x_)
        takestep_history.append(StepRecord(x, ret_x, self.takestep.stepsize, adjusted))
        return ret_x

# ---- Take Step Cumtomization End ---------------------------------------------

# objective function
def objective(v):
    x, y = v
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# define range for input
r_min, r_max = -5.0, 5.0
# define the starting point as a random sample from the domain

minima_list = []
init_points = [(-4, -4), (-4, -4), (4, -4), (4, -4), (4, 4), (4, 4), (-4, 4), (-4, -4)]

def solve(k):

    stepsize = 0.5
    take_step = CustomTakestep(stepsize=0.5)

    # pt = r_min + rand(2) * (r_max - r_min)
    pt = init_points[k]

    takestep_history.append(StepRecord(pt, pt, stepsize, False))

    fv = objective(pt)
    minima_list.append((pt, fv, True))

    def minima_callback(x, fv, accept):
        # print(x, fv, accept)
        minima_list.append((x, fv, accept))

    # perform the basin hopping search
    return basinhopping(objective, pt, take_step=take_step, niter=18, callback=minima_callback)

history_pos = 0
for k in range(len(init_points)):
    result = solve(k)
    # modify the first step
    first_step = takestep_history[history_pos]
    first_step.ret_x = takestep_history[history_pos+1].x
    print("first_step.x=", first_step.x, "first_step.ret_x=", first_step.ret_x)

    # add None to make the history size equal to that of minima_list
    takestep_history.append(None)

    history_pos = len(takestep_history)
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

fig.suptitle("Demonstration of Basin Hopping Algorithm using Ackley function", fontsize=20)
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
trace2, = ax2.plot(*minima_array[0:1,0:2].T, color='red', label="local minima trace")
first_step = takestep_history[0]
patch = first_step.create_patch()
ax2.add_patch(patch)

step_points = np.array([(np.nan, np.nan) if h is None else h.x for h in takestep_history])
DRAW_STEP_TRACE = False

if DRAW_STEP_TRACE:
    step_trace, = ax2.plot(*step_points[0:1].T, "-o", color="green", alpha=0.5)
displace = first_step.create_displine(ax2)

min_init, = ax2.plot(*step_points[0], "o", color="cyan", label="last initialization")
min_vector, = ax2.plot(*step_points[0:1].T, color="cyan", alpha=0.5, label="last minimization")
minimizatin = min_init, min_vector

ax2.legend()

print("len(takestep_history)=", len(takestep_history))
print("len(minima_list)=", len(minima_list))

def reset():
    x, y, z = minima_array[0]
    points.set_data(np.array([x]), np.array([y]))
    points.set_3d_properties(np.array([z]), 'z')
    trace3.set_data(*minima_array[0:1,0:2].T)
    trace3.set_3d_properties(minima_array[0:1,2], 'z')
    point.set_data([x], [y])
    trace2.set_data(*minima_array[0:1, 0:2].T)
    first_step = takestep_history[0]
    first_step.move_patch(patch)
    first_step.move_displine(displace)
    min_init.set_data([x], [y])
    min_vector.set_data(*step_points[0:1].T)
    if DRAW_STEP_TRACE:
        step_trace.set_data(*step_points[0:1].T)
        return points, trace3, point, trace2, patch, step_trace, *displace, *minimizatin
    else:
        return points, trace3, point, trace2, patch, *displace, *minimizatin

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
    this_step = takestep_history[i]
    if this_step is not None:
        this_step.move_patch(patch)
        this_step.move_displine(displace)
        if r > 0:
            x_, y_ = takestep_history[i-1].ret_x
            min_init.set_data([x_], [y_])
            min_vector.set_data([x_, x], [y_, y])
    if DRAW_STEP_TRACE:
        step_trace.set_data(*step_points[slice_].T)
        return points, trace3, point, trace2, patch, step_trace, *displace, *minimizatin
    else:
        return points, trace3, point, trace2, patch, *displace, *minimizatin

anim = animation.FuncAnimation(fig, animate, frames=len(minima_list), blit=True, init_func=reset, interval=500)
import sys
if len(sys.argv) > 1 and sys.argv[1] == "save":
    anim.save("improved-anim.gif")

plt.show()
