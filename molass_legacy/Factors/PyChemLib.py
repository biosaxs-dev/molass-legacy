"""
    PyChemLib.py

    functions borrowed from
    https://pythoninchemistry.org/sim_and_scat/intro

    Copyright (c) 2023, SAXS Team, KEK-PF
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann
mass_of_argon = 39.948 # amu

def lj_force(r, epsilon, sigma):
    """
    Implementation of the Lennard-Jones potential 
    to calculate the force of the interaction.
    
    Parameters
    ----------
    r: float
        Distance between two particles (Å)
    epsilon: float 
        Potential energy at the equilibrium bond 
        length (eV)
    sigma: float 
        Distance at which the potential energy is 
        zero (Å)
    
    Returns
    -------
    float
        Force of the van der Waals interaction (eV/Å)
    """
    return 48 * epsilon * np.power(
        sigma, 12) / np.power(
        r, 13) - 24 * epsilon * np.power(
        sigma, 6) / np.power(r, 7)

def demo_lj_force():
    r = np.linspace(3.5, 8, 100)
    plt.plot(r, lj_force(r, 0.0103, 3.4))
    plt.xlabel(r'$r$/Å')
    plt.ylabel(r'$f$/eVÅ$^{-1}$')
    plt.show()

def demo_lj_force2():
    r = np.linspace(5.9, 15, 100)
    plt.plot(r, lj_force(r, 0.1, 5.99))
    plt.xlabel(r'$r$/Å')
    plt.ylabel(r'$f$/eVÅ$^{-1}$')
    plt.show()

def init_velocity(T, number_of_particles, mass):
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
    R = np.random.rand(number_of_particles) - 0.5
    return R * np.sqrt(Boltzmann * T / (
        mass * 1.602e-19))

def get_accelerations(positions, mass):
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
    accel_x = np.zeros((positions.size, positions.size))
    for i in range(0, positions.size - 1):
        for j in range(i + 1, positions.size):
            r_x = positions[j] - positions[i]
            rmag = np.sqrt(r_x * r_x)
            force_scalar = lj_force(rmag, 0.0103, 3.4)
            force_x = force_scalar * r_x / rmag
            accel_x[i, j] = force_x / mass #eV Å-1 amu-1
            # appling Newton's third law
            accel_x[j, i] = - force_x / mass
    return np.sum(accel_x, axis=0)

def demo_accel():
    mass_of_argon = 39.948 # amu
    accel = get_accelerations(np.array([1, 5, 10]), mass_of_argon)
    print('Acceleration on particle 0 = {:.3e} eV/Åamu'.format(accel[0]))
    print('Acceleration on particle 1 = {:.3e} eV/Åamu'.format(accel[1]))
    print('Acceleration on particle 2 = {:.3e} eV/Åamu'.format(accel[2]))

def update_pos(x, v, a, dt):
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
    return x + v * dt + 0.5 * a * dt * dt

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

def run_md(dt, number_of_steps, initial_temp, x, mass):
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
    positions = np.zeros((number_of_steps, 3))
    v = init_velocity(initial_temp, 3, mass)
    a = get_accelerations(x, mass)
    for i in range(number_of_steps):
        x = update_pos(x, v, a, dt)
        a1 = get_accelerations(x, mass)
        v = update_velo(v, a, a1, dt)
        a = np.array(a1)
        positions[i, :] = x
    return positions

def demo_run_md():
    x = np.array([1, 5, 10])
    sim_pos = run_md(0.1, 10000, 300, x, mass_of_argon)
        
    # %matplotlib inline
    for i in range(sim_pos.shape[1]):
        plt.plot(sim_pos[:, i], '.', label='atom {}'.format(i))
    plt.xlabel(r'Step')
    plt.ylabel(r'$x$-Position/Å')
    plt.legend(frameon=False)
    plt.show()
