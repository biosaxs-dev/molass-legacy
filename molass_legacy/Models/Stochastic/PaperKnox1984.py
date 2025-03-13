"""
    Models.Stochastic.PaperKnox1984.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt

def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    """
    https://stackoverflow.com/questions/26989131/add-cylinder-to-plot
    """
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

def illust_cylindrical():
    from Theory.SpherePlot import sphere_xyz
    fig, axes = plt.subplots(ncols=3, figsize=(15,5),  subplot_kw=dict(projection='3d'))
    fig.suptitle("Illustation of $K_{SEC}$ Estimation in Cylindrical Pores based on J. H. Knox, 1984", fontsize=20)

    R = 0.5

    for ax, r in zip(axes, [0.1, 0.2, 0.3]):
        rho = r/R
        Ksec = (1 - rho)**2
        ax.set_title("R=%g, r=%g\n\n" % (R, r) + r"$K_{SEC} = (\frac{R - r}{R})^2 = (1 - \frac{r}{R})^2 = (1 - \rho)^2$ = %.3g" % Ksec, y=0.95)
        ax.set_axis_off()

        Xc,Yc,Zc = data_for_cylinder_along_z(0.5, 0.5, R, 1)
        ax.plot_surface(Xc, Zc, Yc, alpha=0.3, label="Pore: Radius=R=%.3g" % R)

        Xc,Yc,Zc = data_for_cylinder_along_z(0.5, 0.5, R - r, 1)
        ax.plot_surface(Xc, Zc, Yc, alpha=0.3, label="Particle Center Boundary: Radius=R - r=%.3g" % (R - r))

        x, y, z = sphere_xyz(center=(0.5, 0.0, r), r=r)
        ax.plot_surface(x, y, z, facecolor='C2', alpha=0.3, label="Particles: Radius=r=%.3g" % r)

        x, y, z = sphere_xyz(center=(r, 0.0, 0.5), r=r)
        ax.plot_surface(x, y, z, facecolor='C2', alpha=0.3)

        x, y, z = sphere_xyz(center=(0.5, 0.0, 1-r), r=r)
        ax.plot_surface(x, y, z, facecolor='C2', alpha=0.3)

        ax.legend(loc="lower right")
        ax.view_init(elev=10, azim=-70, roll=0)

    fig.tight_layout()
    plt.show()

def sphere_xyz_alt(center=(0, 0, 0), radius=1, n=100, phi=None, theta=None):
    if phi is None:
        phi = np.linspace(0, np.pi, 100)
    if theta is None:
        theta = np.linspace(0, 2 * np.pi, 100)

    phi, theta = np.meshgrid(phi, theta)
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)
    return x, y, z

def test_sphere_xyz_alt():
    n = 50
    phi = np.linspace(np.pi*0.2, np.pi*0.8, n)
    x, y, z = sphere_xyz_alt(n=n, phi=phi)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.set_axis_off()
    ax.plot_surface(x, z, y, alpha=0.3)
    ax.view_init(elev=10, azim=-70, roll=0)
    fig.tight_layout()
    plt.show()

def illust_spherical():
    from Theory.SpherePlot import sphere_xyz
    fig, axes = plt.subplots(ncols=3, figsize=(15,5),  subplot_kw=dict(projection='3d'))
    fig.suptitle("Illustation of $K_{SEC}$ Estimation in Spherical Pores based on J. H. Knox, 1984", fontsize=20)


    R = 0.5
    r = 0.2
    n = 50
    # phi = np.linspace(np.pi*0.2, np.pi*0.8, n)
    phi = None

    for ax, r in zip(axes, [0.1, 0.2, 0.3]):
        rho = r/R
        Ksec = (1 - rho)**3
        ax.set_title("R=%g, r=%g\n\n" % (R, r) + r"$K_{SEC} = (\frac{R - r}{R})^3 = (1 - \frac{r}{R})^3 = (1 - \rho)^3$ = %.3g" % Ksec, y=0.95)
        ax.set_axis_off()

        x, y, z = sphere_xyz(center=(0.5, 0.5, 0.5), r=R, n=n, v=phi)
        ax.plot_surface(x, z, y, alpha=0.2, label="Pore: Radius=R=%.3g" % R)

        x, y, z = sphere_xyz(center=(0.5, 0.5, 0.5), r=R - r, n=n, v=phi)
        ax.plot_surface(x, z, y, alpha=0.2, label="Particle Center Boundary: Radius=R - r=%.3g" % (R - r))

        x, y, z = sphere_xyz(center=(0.5, r, 0.5), r=r)
        ax.plot_surface(x, z, y, facecolor='C2', alpha=0.2, label="Particles: Radius=r=%.3g" % r)

        x, y, z = sphere_xyz(center=(r, 0.5, 0.5), r=r)
        ax.plot_surface(x, z, y, facecolor='C2', alpha=0.2)

        x, y, z = sphere_xyz(center=(0.5, 1-r, 0.5), r=r)
        ax.plot_surface(x, z, y, facecolor='C2', alpha=0.2)

        ax.legend(loc="lower right")
        ax.view_init(elev=10, azim=-70, roll=0)

    fig.tight_layout()
    plt.show()

def illust_exclusion_coefficient():

    rhov = np.linspace(0.0, 1.0, 100)
    kv2 = np.power(1 - rhov, 2)
    kv3 = np.power(1 - rhov, 3)
    thv = np.arccos(1/(1 + rhov))
    kv_inv_4cyl = (1 - np.sqrt((2*rhov - rhov**2)) - (1-rhov)**2*(np.pi/4 - thv))/(1 - np.pi/4) 

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
    ax2.set_yscale('log')

    for ax in [ax1, ax2]:
        ax.plot(kv2, rhov)
        ax.plot(kv3, rhov)
        ax.plot(kv_inv_4cyl, rhov)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    # import seaborn as sns
    # sns.set_theme()
    sys.path.append("../lib")

    # illust_cylindrical()
    # test_sphere_xyz_alt()
    # illust_spherical()
    illust_exclusion_coefficient()