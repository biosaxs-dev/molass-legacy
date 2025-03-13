"""
    PyChemLib3D.py

    functions borrowed from
    https://pythoninchemistry.org/sim_and_scat/intro

    Copyright (c) 2023, SAXS Team, KEK-PF
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.constants import Boltzmann
mass_of_argon = 39.948 # amu

from .PyChemLib import lj_force

NDIM = 3
BOX_LIMITS = np.array([(-40, 40), (-40, 40), (-40, 40)])

def init_velocity(T, num_particles, mass):
    """
    Initialise the velocities for a series of 
    particles.
    
    Parameters
    ----------
    T: float
        Temperature of the system at 
        initialisation (K)
    number_of_particles: int
        Number of particles in the system
    
    Returns
    -------
    ndarray of floats
        Initial velocities for a series of 
        particles (eVs/Åamu)
    """
    R = np.random.rand(num_particles, NDIM) - 0.5
    return R * np.sqrt(Boltzmann * T / (
        mass * 1.602e-19))

def get_accelerations(positions, mass, epsilon, sigma, closure_radius):
    """
    Calculate the acceleration on each 
    particle as a  result of each other 
    particle. 
    N.B. We use the Python convention of
    numbering from 0.
    
    Parameters
    ----------
    positions: ndarray of floats
        The positions, in a single dimension, 
        for all of the particles (Å)
        
    Returns
    -------
    ndarray of floats
        The acceleration on each particle (eV/Åamu)
    """
    n = positions.shape[0]
    dim = positions.shape[1]
    assert dim == NDIM
    accel_x = np.zeros((n, n, dim))
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            r_x = positions[j] - positions[i]
            rmag = np.sqrt(np.sum(r_x**2))
            force_scalar = lj_force(max(1, rmag - closure_radius), epsilon, sigma)
            force_x = force_scalar * r_x / rmag
            accel_x[i, j] = force_x / mass  # eV Å-1 amu-1
            # appling Newton's third law
            accel_x[j, i] = - force_x / mass
    return np.sum(accel_x, axis=0)

def demo_accel():
    mass_of_argon = 39.948 # amu
    epsilon = 0.0103
    sigma = 3.4
    accel = get_accelerations(np.array([(1,1,1), (5,5,5), (10,10, 10)]), mass_of_argon, epsilon, sigma)
    print('Acceleration on particle 0 = %.3e, %.3e, %.3e eV/Åamu' % tuple(accel[0]))
    print('Acceleration on particle 1 = %.3e, %.3e, %.3e eV/Åamu' % tuple(accel[1]))
    print('Acceleration on particle 2 = %.3e, %.3e, %.3e eV/Åamu' % tuple(accel[2]))

def update_pos(x, v, a, dt, box_limits):
    """
    Update the particle positions.
    
    Parameters
    ----------
    x: ndarray of floats
        The positions of the particles in a 
        single dimension
    v: ndarray of floats
        The velocities of the particles in a 
        single dimension
    a: ndarray of floats
        The accelerations of the particles in a 
        single dimension
    dt: float
        The timestep length
    
    Returns
    -------
    ndarray of floats:
        New positions of the particles in a single 
        dimension
    """
    zzzz = np.zeros(len(x))
    xmin = np.ones(len(x))*box_limits[0][0]
    xmax = np.ones(len(x))*box_limits[0][1]
    ymin = np.ones(len(x))*box_limits[1][0]
    ymax = np.ones(len(x))*box_limits[1][1]
    x_ = x + v * dt + 0.5 * a * dt * dt
    exmin = np.min([x_[:,0] - xmin, zzzz], axis=0)
    exmax = np.max([x_[:,0] - xmax, zzzz], axis=0)
    eymin = np.min([x_[:,1] - ymin, zzzz], axis=0)
    eymax = np.max([x_[:,1] - ymax, zzzz], axis=0)
    ezmin = np.min([x_[:,2] - ymin, zzzz], axis=0)
    ezmax = np.max([x_[:,2] - ymax, zzzz], axis=0)
    ex = exmin + exmax
    ey = eymin + eymax
    ez = ezmin + ezmax
    e_ = np.array([ex, ey, ez]).T
    v[ex!=0,0] *= -1
    v[ey!=0,1] *= -1
    v[ez!=0,2] *= -1
    return x_ - 2*e_

def update_velo(v, a, a1, dt):
    """
    Update the particle velocities.
    
    Parameters
    ----------
    v: ndarray of floats
        The velocities of the particles in a 
        single dimension (eVs/Åamu)
    a: ndarray of floats
        The accelerations of the particles in a 
        single dimension at the previous 
        timestep (eV/Åamu)
    a1: ndarray of floats
        The accelerations of the particles in a
        single dimension at the current 
        timestep (eV/Åamu)
    dt: float
        The timestep length
    
    Returns
    -------
    ndarray of floats:
        New velocities of the particles in a
        single dimension (eVs/Åamu)
    """
    return v + 0.5 * (a + a1) * dt

def run_md(num_particles, dt, number_of_steps, initial_temp, x, mass, box_limits, epsilon, sigma, closure_radius):
    """
    Run a MD simulation.
    
    Parameters
    ----------
    dt: float
        The timestep length (s)
    number_of_steps: int
        Number of iterations in the simulation
    initial_temp: float
        Temperature of the system at 
        initialisation (K)
    x: ndarray of floats
        The initial positions of the particles in a 
        single dimension (Å)
        
    Returns
    -------
    ndarray of floats
        The positions for all of the particles 
        throughout the simulation (Å)
    """
    positions = np.zeros((number_of_steps, num_particles, NDIM))
    v = init_velocity(initial_temp, num_particles, mass)
    a = get_accelerations(x, mass, epsilon, sigma, closure_radius)
    for i in range(number_of_steps):
        x = update_pos(x, v, a, dt, box_limits)
        a1 = get_accelerations(x, mass, epsilon, sigma, closure_radius)
        v = update_velo(v, a, a1, dt)
        a = np.array(a1)
        positions[i, :] = x
    return positions

def generate_ensemble_impl(num_particles, box_limits):
    ndim = len(box_limits)
    points = np.random.uniform(0, 1, (num_particles, ndim))
    for i, (vmin, vmax) in enumerate(box_limits):
        points[:,i] = vmin + points[:,i]*(vmax - vmin)
    return points

def generate_ensemble(num_particles, box_limits, min_dist=None):
    points = generate_ensemble_impl(num_particles, box_limits)
    if min_dist is None:
        return points

    min_dist2 = min_dist**2
    while True:
        too_close = False
        for i in range(num_particles):
            for j in range(i+1, num_particles):
                dist2 = np.sum((points[i] - points[j])**2)
                if dist2 < min_dist2:
                    too_close = True
                    break
            if too_close:
                break
        if too_close:
            print("retrying due to too close", np.sqrt(dist2))
            points = generate_ensemble_impl(num_particles, box_limits)
            continue
        else:
            break

    return points

def compute_min_dist(points):
    num_particles = len(points)
    min_dist2 = None
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            dist2 = np.sum((points[i] - points[j])**2)
            if min_dist2 is None or dist2 < min_dist2:
                min_dist2 = dist2
    return np.sqrt(min_dist2)

def demo_lj():
    epsilon = 0.0103
    sigma = 3.4
    r = np.linspace(3.5, 8, 100)

    fig, ax1 = plt.subplots()
    ax1.set_title("Lennard-Jones Potential")
    ax1.plot(r, lj_force(r, epsilon, sigma))
    ax1.set_xlabel("Distance")
    ax1.set_ylabel("Potential")
    fig.tight_layout()
    plt.show()

def demo_run_md(num_particles):
    epsilon = 0.0103
    sigma = 3.4
    closure_radius = 4.5

    box_limits = BOX_LIMITS
    x = generate_ensemble(num_particles, box_limits, min_dist=6)
    num_particles = len(x)
    sim_pos = run_md(num_particles, 0.1, 20000, 300, x, mass_of_argon, box_limits, epsilon, sigma, closure_radius)
    num_frames = len(sim_pos)//100

    dist_array = np.zeros(num_frames)
    for k in range(num_frames):
        dist_array[k] = compute_min_dist(sim_pos[k*100])
    max_i = np.argmax(dist_array)

    pause = False

    def on_click(event):
        nonlocal pause
        print('on_click')
        if event.inaxes is None:
            return
        pause ^= True

    fig, ax2 = plt.subplots(subplot_kw=dict(projection='3d'))
    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    ax2.set_title("Particle Movement Animation N=%d" % num_particles)
    iter_text = ax2.text2D(0.1, 0.9, "Iter: 0", alpha=0.5, transform=ax2.transAxes)
    dist_text = ax2.text2D(0.75, 0.9, "Dist: _", alpha=0.5, transform=ax2.transAxes)
    point_arts, = ax2.plot(sim_pos[0,:,0], sim_pos[0,:,1], sim_pos[0,:,2], "o")
    ax2.set_xlabel(r'X')
    ax2.set_ylabel(r'Y')
    ax2.set_zlabel(r'Z')
    ax2.set_xlim(*box_limits[0])
    ax2.set_ylim(*box_limits[1])
    ax2.set_zlim(*box_limits[2])
    # ax2.legend()
    fig.tight_layout()

    def index_generator():
        i = 0
        while i < num_frames:
            i_ = i
            if pause:
                pass
            else:
                i += 1
            yield i_

    def init():
        iter_text.set_text("Iter: 0")
        dist_text.set_text("Dist: %.1f" % dist_array[0])
        return point_arts, iter_text, dist_text

    def update(i):
        pos = sim_pos[i*100]
        point_arts.set_data(pos[:,0], pos[:,1])
        point_arts.set_3d_properties(pos[:,2])
        iter_text.set_text("Iter: %d" % i)
        dist_text.set_text("Dist: %.1f" % dist_array[i])
        return point_arts,iter_text, dist_text

    ani = FuncAnimation(fig, update, index_generator,
                        init_func=init, blit=True)

    plt.show()

    ani.save("md.gif")

    print("max min_dist=%.1f" % dist_array[max_i])
    pos = sim_pos[max_i*100]
    np.savetxt("points.dat", pos)
