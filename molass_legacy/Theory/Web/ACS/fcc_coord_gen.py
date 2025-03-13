"""
Computing the 3D Radial Distribution Function from Particle Positions: An Advanced Analytic Approach
Bernd A. F. Kopera and Markus Retsch
Analytical Chemistry 2018 90 (23), 13909-13914
DOI: 10.1021/acs.analchem.8b03157
"""

from numpy import sqrt
from random import random
from numpy . random import normal
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D

"""
This program calculates FCC coordinates.
A 3D Gaussian disturbance can be added.
Random Sphere Point Picking Algorithm from:
G. Marsaglia, The Annals of Mathematical Statistics,
1972, Vol. 43, No. 2, 845-646
"""

### DASHBOARD

SimuName = "FCC_Blurr_Cube" # Name of the Experiment and the Files
GitterKonst = sqrt(2)       # Cell constant for the FCC unit cell
PartDiam = 1                # Diameter of the Particles
Blurr = True                # add a 3D gaussian blurr?
sigma = 0.07                # standard deviation of the blurr
Lx = 4                      # Distance in x-direction from -Lx to Lx
Ly = 4                      # Distance in y-direction from -Ly to Ly
Lz = 4                      # Distance in z-direction from -Lz to Lz

"""***********************************************************"""

def calcFCCCoor(GitterKonst, Lx, Ly, Lz):

    CoorList = []

    XLim = int(Lx/GitterKonst*5)
    YLim = int(Ly/GitterKonst*5)
    ZLim = int(Lz/GitterKonst*5)

    L = GitterKonst/2

    A = [0, L, L]
    B = [L, 0, L]
    C = [L, L, 0]

    for h in range(-ZLim*2, ZLim*2, 1):
        for k in range(-YLim*2, YLim*2, 1):
            for l in range(-XLim*2, XLim*2, 1):

                NewX = h * A[0] + k * B[0] + l * C[0]
                NewY = h * A[1] + k * B[1] + l * C[1]
                NewZ = h * A[2] + k * B[2] + l * C[2]

                if (abs(NewX) < Lx and
                    abs(NewY) < Ly and
                    abs(NewZ) < Lz):

                    CoorList.append([NewX, NewY, NewZ])

    return CoorList

"""***********************************************************"""

def getRandOffset(sigma):

    x1 = 2
    x2 = 2

    while x1**2 + x2**2 >= 1:

        x1 = random()*2-1
        x2 = random()*2-1

    R = normal(0, sigma)

    Wurzel = sqrt(1 - x1**2 - x2**2)
    x = 2 * x1 * Wurzel
    y = 2 * x2 * Wurzel
    z = 1 - 2 * (x1**2 + x2**2)

    return [x * R, y * R, z * R]

"""***********************************************************"""

def addGaussianBlurr(CoorList, sigma):

    BlurrList = []

    for P in CoorList:

        Offset = getRandOffset(sigma)

        NewX = P[0] + Offset[0]
        NewY = P[1] + Offset[1]
        NewZ = P[2] + Offset[2]

        BlurrList.append([NewX, NewY, NewZ])

    return BlurrList

"""***********************************************************"""

def plotCorr(CoorList, SimuName):

    X = [0] * len(CoorList)
    Y = [0] * len(CoorList)
    Z = [0] * len(CoorList)

    for P in range(0, len(CoorList), 1):

        X[P] = CoorList[P][0]
        Y[P] = CoorList[P][1]
        Z[P] = CoorList[P][2]

    fig = figure()
    # ax = fig.add_subplot(111, projection = "3d", aspect="equal") 
    # NotImplementedError: Axes3D currently only supports the aspect argument 'auto'. You passed in 'equal'.
    ax = fig.add_subplot(111, projection = "3d")

    ax.scatter(X, Y, Z)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    savefig(SimuName + "_3D.png", dpi = 600)
    show ( )

    return True

"""***********************************************************"""

def saveData(CoorList, SimuName):

    DataFile = open(SimuName + "_rawCoor.txt", "w")

    for Particle in CoorList:

        DataFile.write( str(Particle[0]) + "\t" +
                        str(Particle[1]) + "\t" +
                        str(Particle[2]) + "\n")

    DataFile.close()

    return True

"""***********************************************************"""

def createAutoCADFile(CoorList, PartDiam, SimuName):

    CADFile = open(SimuName + "_CAD.scr", "w")

    for Particle in CoorList:

        CADFile.write("SPHERE\n" ) # SPHERE command
        CADFile.write(str(round(Particle[0], 5)) + ",")     # X-Position
        CADFile.write(str(round(Particle[1], 5)) + ",")     # Y-Position
        CADFile.write(str(round(Particle[2], 5)) + "\n")    # Z-Position
        CADFile.write(str(PartDiam/2) + "\n")   # Radius

    CADFile.write("\n")     # Blank line at the end
    CADFile.close()

    return True

"""***********************************************************"""

def generate_fcc_coords(blurr=True):
    import numpy as np
    print("Calculating FCC Coordinates.")
    FCC = calcFCCCoor(GitterKonst, Lx, Ly, Lz)
    if blurr:
        print("Adding a Gaussian blurr with Sigma = " + str(sigma))
        FCC = addGaussianBlurr(FCC, sigma)
    return np.array(FCC)

###MAIN
if __name__ == '__main__':

    print("Calculating FCC Coordinates.")
    FCC = calcFCCCoor(GitterKonst, Lx, Ly, Lz)

    print( "Particle Number : " +str(len(FCC)))

    if Blurr == True :

        print("Adding a Gaussian blurr with Sigma = " + str(sigma))
        FCC = addGaussianBlurr(FCC, sigma)

    print("Saveing the Coordinates to a .txt file.")
    saveData (FCC, SimuName )

    print("Creating a AutoCAD file.")
    createAutoCADFile (FCC, PartDiam , SimuName)

    print( "Plotting the Coordinates.")
    plotCorr(FCC, SimuName)
