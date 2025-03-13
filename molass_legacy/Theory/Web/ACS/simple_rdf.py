"""
Computing the 3D Radial Distribution Function from Particle Positions: An Advanced Analytic Approach
Bernd A. F. Kopera and Markus Retsch
Analytical Chemistry 2018 90 (23), 13909-13914
DOI: 10.1021/acs.analchem.8b03157
"""

from scipy import spatial
from numpy import sqrt, pi, zeros, array_equal

"""
This function calculates the pair correlation function g(2)(r)
for a dense set of particles in a rectangular box. The particle
coordinates are supplied as [[x, y, z],...] lists.
All particles are used as central particles.
The spherical shell might extend beyond the known region.
A KDTree data structure is used for efficiency.
"""

"""************************************************************"""

def RDF_Simple(Particles, r, dr):

    # preallocate list for Gr
    Gr = zeros(len(r))

    # maximal radial distance
    MaxDist = r[-1] + dr/2

    # sort all Particles in a KDTree
    ParticleTree = spatial.KDTree(Particles)

    # use every particle as the center once
    k = 0
    for CentralP in Particles:

        # these are the indices for the particles at most MaxDist away:
        NNIndices = ParticleTree.query_ball_point(CentralP,
                                                  MaxDist, p = 2 , eps = 0)

        # look at every other particle at most MaxDist away:
        for Neighbour in NNIndices:

            if not array_equal(CentralP, Particles[Neighbour]):

                # calc the distance to the neighbour
                dx = CentralP[0] - Particles[Neighbour][0]
                dy = CentralP[1] - Particles[Neighbour][1]
                dz = CentralP[2] - Particles[Neighbour][2]

                d = sqrt(dx**2 + dy**2 + dz**2)

                # what bins is the particle in?
                IdxList = [k for k in range(0, len(r), 1 )
                            if abs(r[k] - d) <= dr]

                # add one to every bin the particle is in
                for Pos in IdxList:

                    # count the particle
                    Gr[Pos] += 1
        k += 1
        print("Finished Particle " + str(k) + " of " + str(len(Particles)))

    # final normalization
    Gr[0] = 0

    BoxVol = (max(Particles[0]) - min(Particles[0]))*\
             (max(Particles[1]) - min(Particles[1]))*\
             (max(Particles[2]) - min(Particles[2]))

    for i in range(1, len(r), 1):

        Gr[i] /= len(Particles) # average over all central particles

        # the most inner shells are spheres!
        if r[i] - dr/2 > 0:

            # by the shell volume
            Gr[i] /= 4/3 * pi * ((r[i] + dr/2)**3 - (r[i] - dr/2)**3)

        else:
            Gr[i] /= 4/3 * pi * (r[i] + dr/2)**3    # by the shell volume

        Gr[i] *= BoxVol / len(Particles)    # by the number density

    return Gr

"""************************************************************"""
