"""
    adapted from
        Radial distribution function
        https://pythoninchemistry.org/sim_and_scat/calculating_scattering/rdfs.html
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pylj import md, sample

def md_simulation(number_of_particles, temperature, 
                  box_length, number_of_steps, 
                  sample_frequency):
    """
    Runs a molecular dynamics simulation in using the pylj 
    molecular dynamics engine.
    
    Parameters
    ----------
    number_of_particles: int
        The number of particles in the simulation
    temperature: float
        The temperature for the initialisation and 
        thermostating
    box_length: float
        The length of the simulation square
    number_of_steps: int
        The number of molecular dynamics steps to run
    sample_frequency: 
        How regularly the visualisation should be updated
        
    Returns
    -------
    pylj.util.System
        The complete system information from pylj
    """
    # %matplotlib notebook
    system = md.initialise(number_of_particles, temperature, 
                           box_length, 'square')
    xpos = system.particles["xposition"].copy()
    ypos = system.particles["yposition"].copy()
    x = []
    y = []
    average_rdf = []
    r = []
    system.time = 0
    for i in range(0, number_of_steps):
        system.integrate(md.velocity_verlet)
        system.md_sample()
        system.heat_bath(temperature)
        system.time += system.timestep_length
        system.step += 1
        if system.step % sample_frequency == 0:
            # sample_system.update(system)
            modified_update(system, x, y, average_rdf, r)

    make_animation(system, xpos, ypos, x, y, average_rdf, r)

def modified_update(system, x, y, average_rdf, r):
    x3 = system.particles["xposition"].copy()
    y3 = system.particles["yposition"].copy()
    x.append(x3)
    y.append(y3)
    hist, bin_edges = np.histogram(
        system.distances, bins=np.linspace(0, system.box_length / 2 + 0.5e-10, 100)
    )
    gr = hist / (
        system.number_of_particles
        * (system.number_of_particles / system.box_length ** 2)
        * np.pi
        * (bin_edges[:-1] + 0.5e-10 / 2.0)
        * 0.5
    )
    average_rdf.append(gr)
    x = bin_edges[:-1] + 0.5e-10 / 2
    r.append(x)

def make_animation(system, xpos, ypos, x, y, average_rdf, r, scale = 1):
    gr = np.average(average_rdf, axis=0)
    ar = np.average(r, axis=0)

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,6))
    fig.suptitle("MD Simulation Demo using pylj @ https://pythoninchemistry.org/", fontsize=20)
    ax1.set_title("Particle Positions", fontsize=16)
    ax2.set_title("Snapshot RDF", fontsize=16)
    ax3.set_title("Averaged RDF", fontsize=16)

    mk = 6.00555e-8 / (system.box_length - 2.2727e-10) - 1e-10 / scale
    art1, = ax1.plot(xpos, ypos, "o", markersize=mk, markeredgecolor="black", color="#34a5daff")
    ax1.set_xlim([0, system.box_length])
    ax1.set_ylim([0, system.box_length])
    ax1.set_xticks([])
    ax1.set_yticks([])

    for ax in [ax2, ax3]:
        ax.set_xlim([0, system.box_length / 2])
        # ax.set_yticks([])
        ax.set_ylabel("RDF", fontsize=16)
        ax.set_xlabel("r/m", fontsize=16)

    art2, = ax2.plot([0], color="#34a5daff")
    ax3.plot(ar, gr, color="#34a5daff")
    ymin, ymax = ax3.get_ylim()
    ymax *= 1.2
    for ax in [ax2, ax3]:
        ax.set_ylim(ymin, ymax)

    def init():
        return art1, art2

    def update(i):
        art1.set_data(x[i], y[i])
        gr = average_rdf[i]
        art2.set_data(r[i], gr)
        return art1, art2

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)

    num_frames = len(x)
    print("num_frames=", num_frames)
    ani = FuncAnimation(fig, update, frames=num_frames,
                        init_func=init, blit=True)
    plt.show()

    # ani.save('pylj-md_simulation.mp4', writer="ffmpeg")
    ani.save('pylj-md_simulation.gif')

if __name__ == '__main__':
    md_simulation(20, 300, 20, 2000, 25)
