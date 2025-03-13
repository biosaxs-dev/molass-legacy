"""
    PyChemLib2D.py

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

NDIM = 2

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

def get_accelerations(positions, mass, epsilon, sigma):
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
            force_scalar = lj_force(rmag, epsilon, sigma)
            force_x = force_scalar * r_x / rmag
            accel_x[i, j] = force_x / mass #eV Å-1 amu-1
            # appling Newton's third law
            accel_x[j, i] = - force_x / mass
    return np.sum(accel_x, axis=0)

def demo_accel():
    mass_of_argon = 39.948 # amu
    epsilon = 0.0103
    sigma = 3.4
    accel = get_accelerations(np.array([(1,1), (5,5), (10,10)]), mass_of_argon, epsilon, sigma)
    print('Acceleration on particle 0 = %.3e, %.3e eV/Åamu' % tuple(accel[0]))
    print('Acceleration on particle 1 = %.3e, %.3e eV/Åamu' % tuple(accel[1]))
    print('Acceleration on particle 2 = %.3e, %.3e eV/Åamu' % tuple(accel[2]))

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
    ex = exmin + exmax
    ey = eymin + eymax
    e_ = np.array([ex, ey]).T
    v[ex!=0,0] *= -1
    v[ey!=0,1] *= -1
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

def run_md(num_particles, dt, number_of_steps, initial_temp, x, mass, box_limits, epsilon, sigma):
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
    a = get_accelerations(x, mass, epsilon, sigma)
    for i in range(number_of_steps):
        x = update_pos(x, v, a, dt, box_limits)
        a1 = get_accelerations(x, mass, epsilon, sigma)
        v = update_velo(v, a, a1, dt)
        a = np.array(a1)
        positions[i,:] = x
    return positions

def demo_run_md():
    epsilon = 0.0103
    sigma = 3.4

    box_limits = np.array([(-10, 20), (-10, 20)])
    x = np.array([(0,0), (5,5), (5,0), (0, 5), (10,10), (5,-9), (-9,5)])
    num_particles = len(x)
    sim_pos = run_md(num_particles, 0.1, 10000, 300, x, mass_of_argon, box_limits, epsilon, sigma)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))

    r = np.linspace(3.5, 8, 100)
    ax1.set_title("Lennard-Jones Potential")
    ax1.plot(r, lj_force(r, epsilon, sigma))
    ax1.set_xlabel("Distance")
    ax1.set_ylabel("Potential")

    ax2.set_title("Particle Movement Animation")
    point_arts, = ax2.plot(sim_pos[0,:,0], sim_pos[0,:,1], "o")
    ax2.set_xlabel(r'X')
    ax2.set_ylabel(r'Y')
    ax2.set_xlim(-10, 20)
    ax2.set_ylim(-10, 20)
    # ax2.legend()
    fig.tight_layout()

    def init():
        return point_arts,

    def update(i):
        pos = sim_pos[i*100]
        point_arts.set_data(pos[:,0], pos[:,1])
        return point_arts,

    ani = FuncAnimation(fig, update, frames=len(sim_pos)//100,
                        init_func=init, blit=True)

    plt.show()

    # ani.save("md.gif")
