"""
Computing the 3D Radial Distribution Function from Particle Positions: An Advanced Analytic Approach
Bernd A. F. Kopera and Markus Retsch
Analytical Chemistry 2018 90 (23), 13909-13914
DOI: 10.1021/acs.analchem.8b03157
"""

from numpy import sqrt, pi, zeros, ones, arctan
# from random import random, seed
# from time import time
# seed(time())

"""
This function calculates the pair correlation functrion g(2)(r)
for a dense set of particles in a rectanglar box. The partricle
coordinates are supplied as [[x, y, z],...] lists.
All particles are used as central aprticles.
The spherical shell might extend beyond the known region.
An empty surrounding is assumed.
"""

"""***********************************************************"""

def SphereCutVol(Rs, A, B):

    Root = sqrt(Rs**2-A**2-B**2)

    Vcut = 1/6*Rs**3*(pi - 2*arctan(A*B/(Rs*Root)))
    Vcut += 1/2 * (arctan(A/Root) - pi/2)*(Rs**2*B-1/3*B**3)
    Vcut += 1/2 * (arctan(B/Root) - pi/2)*(Rs**2*A-1/3*A**3)
    Vcut += 1/3 * A * B * Root

    return Vcut

"""***********************************************************"""

def OctVolume(Rs, xb, yb, zb):

    # if all boundaries are fully in the octant
    if xb**2 + yb**2 + zb **2 < Rs**2:

        return xb * yb * zb

    # if no boundary intersects we start with
    VOctant = 1/8 * 4/3 * pi * Rs**3

    # remove the spherical caps
    for B in [xb , yb, zb]:

        if B < Rs:

            VOctant -= pi/4*(2/3*Rs**3-B*Rs**2+1/3*B**3)

    # add the intersections of the caps
    for (a, b) in [(xb, yb), (xb, zb), (yb ,zb)]:

        if a**2 + b**2 < Rs**2:

            VOctant += SphereCutVol(Rs, a, b)

    return VOctant

"""**********************************************************"""

def SphereVolume(Rs, BoxBounds):

    [Xmin, Xmax, Ymin, Ymax, Zmin, Zmax] = BoxBounds

    VSphere = 0

    # abs() mirrors the boundaries into the first octant
    for xb in [Xmin, Xmax]:
        for yb in [Ymin ,Ymax]:
            for zb in [Zmin, Zmax]:

                VSphere += OctVolume(Rs, abs(xb), abs(yb), abs(zb))

    return VSphere

"""***********************************************************"""

def ShellVolume(Rmin, Rmax, BoxBounds):

    # check for negative Rmin values
    Rmin = max([Rmin, 0])

    InnerShell = SphereVolume(Rmin, BoxBounds)
    OuterShell = SphereVolume(Rmax, BoxBounds)

    Volume = OuterShell - InnerShell

##  if Volume <= 0:
##      print("Volume = " + str(Volume))
##      print("Rmin = " + str(Rmin))
##      print("Rmax = " + str(Rmax))
##      print("BoxBounds: " + str(BoxBounds))

    return Volume

"""************************************************************"""

def RDF_AnalyticNorm(Particles, r, dr):

    # Gr averaged over all particles
    Global_Gr = zeros(len(r))

    # Keep track of the use full shell volumes
    NonEmptyShells = zeros(len(r))

    # maximal radial distance
    MaxDist = r[-1] + dr/2

    # Box boundaries
    XList = [Particles[k][0] for k in range(0, len(Particles), 1)]
    YList = [Particles[k][1] for k in range(0, len(Particles), 1)]
    ZList = [Particles[k][2] for k in range(0, len(Particles), 1)]

    BoxBounds =[min(XList), max(XList),
                min(YList), max(YList),
                min(ZList), max(ZList)]

    # box size
    Lx = BoxBounds[1] - BoxBounds [0]
    Ly = BoxBounds[3] - BoxBounds [2]
    Lz = BoxBounds[5] - BoxBounds [4]

    print("Lx = " + str(Lx))
    print("Ly = " + str(Ly))
    print("Lz = " + str(Lz))

    MeanDensity = len(Particles) / (Lx * Ly * Lz)

    # use every particle as the center once
    for CentralP in range (0, len(Particles), 1):

        # local Gr around the current particle
        Local_Gr = zeros(len(r))

        # look at every other particle at most MaxDist away:
        for Neighbour in range(0, len(Particles), 1):

            if CentralP != Neighbour :

                # calc the distance to the neighbour
                dx = Particles[CentralP][0] - Particles[Neighbour][0]
                dy = Particles[CentralP][1] - Particles[Neighbour][1]
                dz = Particles[CentralP][2] - Particles[Neighbour][2]

                d = sqrt(dx**2 + dy**2 + dz**2)

                # what bins is the particle in?
                IdxList = [k for k in range(0, len(r), 1) if abs (r[k] - d) <= dr/ 2]

                # add one to every bin the particle is in
                for Pos in IdxList:

                    # count the particle
                    Local_Gr[Pos] += 1

        # shift the center of box cosy
        LocalBox = [BoxBounds[0] - Particles[CentralP][0],
                    BoxBounds[1] - Particles[CentralP][0],
                    BoxBounds[2] - Particles[CentralP][1],
                    BoxBounds[3] - Particles[CentralP][1],
                    BoxBounds[4] - Particles[CentralP][2],
                    BoxBounds[5] - Particles[CentralP][2]]

        # normalize with the shell volume
        for RIdx in range(0, len(r), 1):

            SVolume = ShellVolume(r[RIdx]-dr/2, r[RIdx]+dr/2, LocalBox)

            if SVolume > 0.0:

                Local_Gr[RIdx] /= SVolume
                NonEmptyShells[RIdx] += 1

        # normalize by the mean particle density
        Local_Gr = Local_Gr / MeanDensity

        # save in the global gr for the average over particles
        Global_Gr = Global_Gr + Local_Gr

        print("Finished Particle " + str(CentralP) + " of " + str(len(Particles)))

    # final normalization considering the non empty shell volumes

    for k in range(0, len(Global_Gr), 1):

        if NonEmptyShells[k] != 0:

            Global_Gr[k] /= NonEmptyShells[k]

        else:
            print("All Shells at R = " + str(r[k]) + " are Empty!" )

    return Global_Gr

"""************************************************************"""
