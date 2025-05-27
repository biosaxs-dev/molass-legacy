"""
    Theory.RDF.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

def demo_mdtraj():
    import mdtraj as md
    traj = md.load(r'D:\PyTools\Beam\1l36.pdb')
    print(traj.xyz.shape)

    pairs = traj.top.select_pairs('all', 'all')
    r, rdf_all = md.geometry.rdf.compute_rdf(traj, pairs)
    print(rdf_all[-5:])

    fig = plt.figure(figsize=(14,7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    x, y, z = traj.xyz[0].T
    ax1.scatter(x, y, z)
    ax2.plot(r, rdf_all)

    fig.tight_layout()
    plt.show()

def demo2():
    domain_size = 20.0
    num_particles = 100

    particle_radius = 0.1
    rMax = domain_size / 4
    x = np.random.uniform(low=0, high=domain_size, size=num_particles)
    y = np.random.uniform(low=0, high=domain_size, size=num_particles)
    z = np.random.uniform(low=0, high=domain_size, size=num_particles)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z)

    fig.tight_layout()
    plt.show()
