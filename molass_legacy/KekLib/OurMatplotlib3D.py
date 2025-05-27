"""
    OurMatplotlib3D.py

    modified from
    https://stackoverflow.com/questions/18228966/how-can-matplotlib-2d-patches-be-transformed-to-3d-with-arbitrary-normals/18228967

    Copyright (c) 2018-2020, SAXS Team, KEK-PF
"""
import numpy                as np
import matplotlib.pyplot    as plt
from mpl_toolkits.mplot3d import proj3d, art3d
from matplotlib.patches import Ellipse, Polygon
from itertools import product, combinations
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def rotation_matrix(d):
    """
    Calculates a rotation matrix given a vector d. The direction of d
    corresponds to the rotation axis. The length of d corresponds to 
    the sin of the angle of rotation.

    Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    """
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        return np.identity(3)

    d /= sin_angle

    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[    0,  d[2],  -d[1]],
                  [-d[2],     0,  d[0]],
                  [d[1], -d[0],    0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M

def pathpatch_2d_to_3d(pathpatch, z = 0, normal = 'z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0,0,0), index)

    normal /= np.linalg.norm(normal) #Make sure the vector is normalised

    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color    

    verts = path.vertices #Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1)) #Obtain the rotation vector    
    M = rotation_matrix(d) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])

def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta

def simply_to_3d(pathpatch, verts):
    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()
    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color    
    pathpatch._segment3d = np.array(verts)

def plot_parallelogram3d(ax, p0, p1, p2, **kwargs):
    v1 = p1 - p0
    v2 = p2 - p0
    p3 = p0 + v1 + v2
    parallel = Polygon([(0,0), (1,0), (1,1), (0,1)], **kwargs)
    ax.add_patch(parallel)
    simply_to_3d(parallel, [p0, p1, p3, p2, p0])
    return parallel

def demo_pathpatch_2d_to_3d():
    ax = plt.axes(projection = '3d') #Create axes

    p = Ellipse((0,0), .2, .5) #Add an ellipse in the yz plane
    ax.add_patch(p)
    pathpatch_2d_to_3d(p, z = 0.5, normal = 'x')
    pathpatch_translate(p, (0, 0.5, 0))

    p = Ellipse((0,0), .2, .5, facecolor = 'r') #Add an ellipse in the xz plane
    ax.add_patch(p)
    pathpatch_2d_to_3d(p, z = 0.5, normal = 'y')
    pathpatch_translate(p, (0.5, 1, 0))

    p = Ellipse((0,0), .2, .5, facecolor = 'g') #Add an ellipse in the xy plane
    ax.add_patch(p)
    pathpatch_2d_to_3d(p, z = 0, normal = 'z')
    pathpatch_translate(p, (0.5, 0.5, 0))

    for normal in product((-1, 1), repeat = 3):
        p = Ellipse((0,0), .2, .5, facecolor = 'y', alpha = .2)
        ax.add_patch(p)
        pathpatch_2d_to_3d(p, z = 0, normal = normal)
        pathpatch_translate(p, 0.5)

    plt.show()

class Inset2Din3D:
    def __init__(self, ax, inset_box):
        self.ax = ax
        self.axins = axins = ax.inset_axes(inset_box)
        axins.get_xaxis().set_visible(False)
        axins.get_yaxis().set_visible(False)

        x, y, w, h = inset_box
        b1 = ax.transAxes.transform((x, y+h))
        b2 = ax.transAxes.transform((x+w, y))

        inv = ax.transData.inverted()
        c1 = inv.transform(b1)
        c2 = inv.transform(b2)

        self.c_points = [c1, c2]
        self.annotations = []
        self.drawing = False

    def get_axis(self):
        return self.axins

    def set_event_handler(self):
        self.ax.figure.canvas.mpl_connect( 'draw_event', self.draw )

    def set_annotation_lines(self, p1, p2):
        self.a_points = [p1, p2]

    def draw(self, *args):
        ax = self.ax
        M = ax.get_proj()

        for an in self.annotations:
            an.remove()

        self.annotations = []
        for p, c in zip(self.a_points, self.c_points):
            tx, ty, _ = proj3d.proj_transform(*p, M)
            an = ax.annotate( None, xy=(tx, ty), xytext=c,
                            arrowprops=dict( headwidth=0, width=0.01, facecolor='black', shrink=0.01, alpha=0.5),
                            )
            self.annotations.append(an)

def inset_3d_demo():
    import molass_legacy.KekLib.DebugPlot as dplt

    fig = dplt.figure()
    ax = fig.add_subplot(111, projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    ax.plot(x, y, z, label='parametric curve')
    ax.legend(loc='upper left')

    n = 10
    px = x[n:n+1]
    py = y[n:n+1]
    pz = z[n:n+1]
    ax.plot(px, py, pz, 'o', color='red')

    inset = Inset2Din3D(ax, [0.6, 0.7, 0.4, 0.3])
    p1 = p2 = (px[0], py[0], pz[0])
    inset.set_annotation_lines(p1, p2)
    inset.draw()
    inset.set_event_handler()

    fig.tight_layout()
    dplt.show()

def parallelogram(origin, v1, v2):
    return [origin, origin+v1, origin+v1+v2, origin+v2]

def plot_parallelepiped(ax, origin, vectors, color=None, edgecolor='w', alpha=1):
    verts = []
    for i,j,k in [(0,1,2), (1,2,0), (2,0,1)]:
        vi = np.array(vectors[i])
        vj = np.array(vectors[j])
        verts.append(parallelogram(np.zeros(3), vi, vj))
        vk = np.array(vectors[k])
        verts.append(parallelogram(vk, vi, vj))
    ax.add_collection3d(Poly3DCollection(verts, color=color, linewidths=1, alpha=alpha))
    ax.add_collection3d(Line3DCollection(verts, colors=edgecolor, linewidths=0.5, linestyles=':'))

def plot_cube(ax, origin, r, **kwargs):
    pass

def plot_sphere(ax, center, r, n=100, **kwargs):
    """
        [matplotlib animation] 3.Z-axis rotation animation of the sphere
        https://sabopy.com/en/matplotlib-animation-3/
    """
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)

    x, y, z = center
    x_ = r * np.outer(np.cos(u), np.sin(v)) + x
    y_ = r * np.outer(np.sin(u), np.sin(v)) + y
    z_ = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z

    ax.plot_surface(x_, y_, z_, **kwargs)
