from scipy import optimize
from matplotlib import pylab as plt
import numpy as np
import pdb
from numpy import log

def exp_growth(t, x0, r):
    return x0 * ((1 + r) ** t)

def doubling_time(m, x_pts, y_pts):
    window = 10

    x1 = x_pts[m]
    y1 = y_pts[m]
    x2 = x_pts[m+window]
    y2 = y_pts[m+window]

    return (x2 - x1) * log(2) / log(y2 / y1)

# First, artificially create data points to work with
data_points = 42

# Create the x-axis
x_pts = range(0, data_points)

# Create noisy points with: y = x^2 + noise, with unique possible errors
y_pts = []
y_err = []
for i in range(data_points):
    random_scale = np.random.random()
    y_pts.append((i * i) + data_points * random_scale)
    y_err.append(random_scale * 100 + 100)

x_pts = np.array(x_pts)
y_pts = np.array(y_pts)
y_err = np.array(y_err)

# Fit to function
[x0, r], pcov  = optimize.curve_fit(exp_growth, x_pts, y_pts, p0=(0.001, 1.0))
fitted_data = exp_growth(x_pts, x0, r)

# Find doubling times
x_t2 = range(32)
t2 = []
t2_fit = []
for i in range(32):
    t2.append(doubling_time(i, x_pts, y_pts))
    t2_fit.append(doubling_time(i, x_pts, fitted_data))

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

ax1.plot(x_pts, y_pts, 'bo')
ax1.errorbar(x_pts, y_pts, yerr=y_err)
ax1.set_ylim([0, 2000])
ax1.set_title('Artificially created raw data points with unique errors', fontsize=8)

ax2.plot(fitted_data, 'g-')
ax2.set_ylim([0, 2000])
ax2.set_title('Fitted exponential function', fontsize=8)

ax3.plot(x_t2, t2, 'ro', label='From points')
ax3.plot(x_t2, t2_fit, 'bo', label='From fitted')
ax3.set_title('Doubling time at each point (10 unit window)', fontsize=8)
ax3.legend(fontsize='8')

plt.show()
