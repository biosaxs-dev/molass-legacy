"""
    RdfDemo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from bisect import bisect_right
from Theory.Web.ACS.fcc_coord_gen import generate_fcc_coords
from Theory.Web.ACS.analytic_rdf import RDF_AnalyticNorm
from .PyChemLib3D import BOX_LIMITS

def generate_sphere(r=4):
    particles = generate_fcc_coords(blurr=False)

    x = particles[:,0]
    y = particles[:,1]
    z = particles[:,2]
    sphere = x**2 + y**2 + z**2 < r**2
    return particles[sphere,:]

def make_shifted(displ, particles):
    ret_particles = particles.copy()
    for j in range(3):
        ret_particles[:,j] += displ[j]
    return ret_particles

def genarate_params(n):
    radii = np.random.normal(15, 1, n)
    print("radii=", radii)
    displs = np.random.uniform(-1,1,(n,3))
    print("displs=", displs)

    centers = [np.zeros(3)]
    for k in range(n):
        dx, dy, dz = displs[k,:]
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        ratio = radii[k]/r
        centers.append(displs[k,:]*ratio)

    return radii, displs, centers

def generate_molecules(particles1, n, min_dist=9):
    while True:
        radii, displs, centers = genarate_params(n)
        exists_near_pair = False
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < min_dist:
                    exists_near_pair = True
                    break
        if exists_near_pair:
            print("exists_near_pair continue", dist)
            continue
        else:
            print("no near_pairs. ok")
            break

    molecules = [particles1]
    for k in range(n):
        dx, dy, dz = displs[k,:]
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        ratio = radii[k]/r
        molecule = make_shifted(displs[k,:]*ratio, particles1)
        molecules.append(molecule)

    return molecules

"""
radii= [15.00525968 15.49340332 12.27811157 15.75436373]
displs= [[-0.97795202 -0.58386681  0.92010054]
 [ 0.32175131 -0.85231832 -0.46558907]
 [ 0.60603154  0.46890646  0.31046006]
 [ 0.49491593  0.63570313 -0.31370224]]
"""

def demo(center_pos_file=None, rdf_files=None):
    particles1 = generate_sphere()
    print(particles1.shape)

    if center_pos_file is None:
        num_molecules = 8
        num_shifts = num_molecules - 1
        molecules = generate_molecules(particles1, num_shifts)
    else:
        centers = np.loadtxt(center_pos_file)
        molecules = []
        for center in centers:
            molecule = make_shifted(center, particles1)
            molecules.append(molecule)
        num_molecules = len(molecules)

    all_particles = np.concatenate(molecules)
    print("all_particles.shape=", all_particles.shape)
    if rdf_files is None:
        r = np.linspace(0, 60, 200)
        gr_one = RDF_AnalyticNorm(molecules[0], r, 0.05)
        gr_all = RDF_AnalyticNorm(all_particles, r, 0.05)
        np.savetxt("rdf_one.dat", np.array([r, gr_one]).T)
        np.savetxt("rdf_all.dat", np.array([r, gr_all]).T)
    else:
        rdf_one = np.loadtxt(rdf_files[0])
        rdf_all = np.loadtxt(rdf_files[1])
        assert rdf_one.shape == rdf_all.shape
        r = rdf_one[:,0]
        gr_one = rdf_one[:,1]
        gr_all = rdf_all[:,1]

    gs = GridSpec(3,2)
    fig = plt.figure(figsize=(16,8))
    fig.suptitle("Intera/inter-particle Radial Distribution Demo for N=%d" % num_molecules, fontsize=20)

    ax1 = fig.add_subplot(gs[:,0], projection="3d")
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,1])
    ax4 = fig.add_subplot(gs[2,1])

    ax1.set_title("3D View", fontsize=16)

    for particles in molecules:
        ax1.plot(*particles.T, "o", markersize=3, alpha=0.5)

    ax1.set_xlim(*BOX_LIMITS[0])
    ax1.set_ylim(*BOX_LIMITS[1])
    ax1.set_zlim(*BOX_LIMITS[2])

    k = bisect_right(r, 50)
    plot_slice = slice(0, k)
    r_ = r[plot_slice]

    ax2.set_title("RDF of one molecule", fontsize=16)
    ax2.plot(r_, gr_one[plot_slice])
    ax3.set_title("RDF of all molecules", fontsize=16)
    ax3.plot(r_, gr_all[plot_slice])
    ax4.set_title("RDF of all molecules (zoomed)", fontsize=16)
    ax4.plot(r_, gr_all[plot_slice])
    ymin, ymax = ax4.get_ylim()
    ax4.set_ylim(-0.1, 2)

    fig.tight_layout()
    plt.show()

def demo_lj():
    num_molecules = 10
    num_shifts = num_molecules - 1

    radii = np.random.normal(15, 1, num_shifts)
    print("radii=", radii)
    displs = np.random.uniform(-1,1,(num_shifts,3))
    print("displs=", displs)

    points = [(0.0, 0.0, 0.0)]

    for k in range(num_shifts):
        dx, dy, dz = displs[k,:]
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        ratio = radii[k]/r
        points.append(displs[k,:]*ratio)

    points = np.array(points)

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"), figsize=(8,8))
    point_arts, = ax.plot(*points.T, "o")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-20, 20)
    fig.tight_layout()

    pos_list = []
    points_ = points.copy()
    for n in range(10):
        for k in range(points.shape[1]):
            points_[:,k] += 0.1
        pos_list.append(points_.copy())

    def init():
        return point_arts,

    def update(i):
        pos = pos_list[i]
        point_arts.set_data(pos[:,0], pos[:,1])
        point_arts.set_3d_properties(pos[:,2])
        return point_arts,

    ani = FuncAnimation(fig, update, frames=len(pos_list),
                        init_func=init, blit=True)

    plt.show()
