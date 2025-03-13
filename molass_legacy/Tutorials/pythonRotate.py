'''
=====================================
    Rotating 3D voxel animation
=====================================

from https://gist.github.com/mtesseracted/62ba9204d5fd3bb952c830c923c1b321

Demonstrates using ``ax.voxels`` with uneven coordinates
'''
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as manimation
from math import copysign

# Define the letters in a doc string, hashes will be 
#  colored in and the rotation order is alphabetical,
#  starting at A. The letters on the corners show
#  where the top left corner will be when viewing
#  that face.
faceStr="""
     ______
     |#   |
     |####|
     |####|
     C#   |
A====B====F====D=====
|####|#  #|#  #|#  #|
|#  #|#  #|## #|#  #|
|####|####|####|####|
|#   |   #|# ##|#  #|
|#   |####|#  #|#  #|
=====================
          | ## |
          |#  #|
          |#  #|
          | ## |
          E=====
"""

# Box colors
boxcol  = '#909cd540'
edgecol = '#7D84A6'
# 739ec040
# 7a88cc
# Set the color of each face in RGBA hex, in rotation order
fcol1 = '#ffd030d0'
faceCols=[fcol1]*6 # set them all the same


def explode(data):
    '''
    Make a new grid where the original grid is every other point
    in the new grid, starting at the third point.  Also add an
    extra padding row/col when a dimension is even to prevent
    gaps in the middle when viewing that dimension face on.

    Parameters
    ----------
    data: 3D np.array of coordinates
    '''

    size = np.array(data.shape)
    dmxx, dmxy, dmxz = (size * 2) + 1  # max dimension values

    # add extra row/col if even and then double
    size = np.array([x+1 if (x % 2) == 0 else x for x in size])*2

    # make and fill the exploded data
    data_e = np.zeros(size + 1, dtype=data.dtype)
    data_e[2:dmxx:2, 2:dmxy:2, 2:dmxz:2] = data

    return data_e


def voxel_face(corns, dm, nf, offst=0.0):
    '''
    Grab the corner coordinates of one voxel face

    Parameters
    ----------
    corns : np.indices array of corners for one voxel
    dm : (dimension), values can be  0(x), 1(y), 2(z)
    nf : (near/far face), values can be 0(near), 1(far)
    offst: how much to offset the face size, also offsets at
           1/10th that amount in the perpendicular direction
    '''

    lc = corns.copy()  # local copy so we don't swap original
    if dm == 1:  # swap y into x and correct ordering
        lc[0], lc[1] = corns[1].transpose(1, 0, 2), corns[0].transpose(1, 0, 2)
    if dm == 2:  # swap z into x and correct ordering
        lc[0], lc[2] = corns[2].transpose(2, 1, 0), corns[0].transpose(2, 1, 0)

    ret = np.zeros((3, 2, 2))
    xc1 = lc[0, nf, 0, 0]  # hold x dim constant
    xc1 += offst * (2*nf - 1) / 10.0  # offset a little
    ret[0, :] = np.array([[xc1, xc1], [xc1, xc1]])
    yc1, yc2 = lc[1, 0, 0:2, 0]
    yc1 += offst  # shrink to show voxel edges
    yc2 -= offst
    ret[1, :] = np.array([[yc1, yc2], [yc1, yc2]])
    zc1, zc2 = lc[2, 0, 0, 0:2]
    zc1 += offst  # shrink to show voxel edges
    zc2 -= offst
    ret[2, :] = np.array([[zc1, zc1], [zc2, zc2]])

    if dm != 0:  # swap x back into desired dimension
        ret[0], ret[dm] = ret[dm].copy(), ret[0].copy()
    return ret


def create_face(voxels, side, facestr):
    '''
    Change voxels to True for cubes corresponding to
    '#' symbols in facestr on the side specified

    Parameters
    ----------
    voxels: 3D array of bools
    side: string representing the constant face
    facestr: string with '#' representing true and
             rows seperated by newlines
    '''

    sides = {
        #    (start, for_x, for_y)
        'A': ([0, 0, -1], [1, 0, 0], [0, 0, -1]),    # y0
        'B': ([-1, 0, -1], [0, 1, 0], [0, 0, -1]),   # x1
        'C': ([0, 0, -1], [0, 1, 0], [1, 0, 0]),   # z1
        'D': ([0, -1, -1], [0, -1, 0], [0, 0, -1]),  # x0
        'E': ([0, 0, 0], [1, 0, 0], [0, 1, 0]),   # z0
        'F': ([-1, -1, -1], [-1, 0, 0], [0, 0, -1]), # y1
    }
    #'B': ([-1, 0, -1], [0, 1, 0], [0, 0, -1]),   # x1
    #'B': ([-1, -1, -1], [0, -1, 0], [0, 0, -1]),   # x1
    start, for_x, for_y = np.array(sides.get(side))

    for out_y, line in enumerate(facestr.strip('\r\n').splitlines()):
        #for out_x, char in enumerate(line.strip()):
        for out_x, char in enumerate(line):

            x, y, z = start + for_x*out_x + for_y*out_y
            voxels[x, y, z] = (char == '#')

    return



sarr = faceStr.strip('\r\n').split('\n')
lins = [None]*len(sarr)
for i in range(len(sarr)):
    lins[i] = sarr[i].strip(' ')
    #np.fromiter(sarr[i].strip(' '), dtype='S1')

iinds = np.zeros((6,2), dtype=int) #upper and lower lims, alphabetical
jinds = np.zeros((6,2), dtype=int) #vertical lims

# Find the index bounds from faceStr, hard-coded logic
# Key:: A:0, B:1, C:2, D:3, E:4, F:5
#C
iinds[2, 0] = jinds[2, 0] = 0 #C is top corner
iinds[2, 1] = len(lins[0]) -1 #C width is the first line
jinds[2, 1] = [x[0] for x in lins].index('A') #goes until A
#A
iinds[0, 0] = 0 #A is far left
jinds[0, 0] = jinds[2,1] #A_j starts at C end
jinds[0, 1] = [x[0] for x in lins].index('=') #A_j goes until E
iinds[0, 1] = lins[jinds[0, 0]].index('B') #A goes until B
#B
iinds[1, 0] = iinds[0, 1] #B_i starts at A end
iinds[1, 1] = iinds[1, 0] + (iinds[2, 1] - iinds[2, 0]) #same width as C
jinds[1, :] = jinds[0, :] #B_j same as A
#F
iinds[5, 0] = iinds[1, 1] 
iinds[5, 1] = iinds[5, 0] + (iinds[0, 1] - iinds[0, 0]) #same width as A
jinds[5, :] = jinds[0, :] #F_j same as A
#D
iinds[3, 0] = iinds[5, 1] 
iinds[3, 1] = iinds[3, 0] + (iinds[2, 1] - iinds[2, 0]) #same width as C
jinds[3, :] = jinds[0, :] #D_j same as A

#E
iinds[4, :] = iinds[0, :] #E_i same as F
jinds[4, 0] = jinds[0, 1] #top E == bottom A
jinds[4, 1] = jinds[4, 0] + (iinds[2, 1] - iinds[2, 0] ) #Same height as C

jinds[:, 1] -= 1 # shave off bottom '='
#'''

# build letters, @TODO: change hard-code dims
#n_voxels = np.zeros((4, 4, 5), dtype=bool)
ylen = iinds[2,1] - iinds[2,0] - 1
xlen = iinds[0,1] - iinds[0,0] - 1
zlen = jinds[0,1] - jinds[0,0] 
n_voxels = np.zeros((xlen, ylen, zlen), dtype=bool)

letters = [None]*6
lett_face = np.zeros((6, 2), dtype=int)

jinds[:, 1] -= 1 #shave off bottom '='
for i in range(6): #len(sarr)):
    i1, i2 = iinds[i, :]
    j1, j2 = jinds[i, :]
    istr = "\n"
    for j in range(j2-j1+1):
        istr += lins[j1+j+1][i1+1:i2] + '\n'
        #print(lins[j1+j+1][i1+1:i2])

    create_face(n_voxels, chr(ord('A') + i), istr)
    letters[i] = np.array(np.where(n_voxels)).T
    n_voxels[...] = False

lett_face[0] = [1, 0]  # close y face
lett_face[1] = [0, 1]  # far x face 
lett_face[2] = [2, 1]  # far z face
lett_face[3] = [0, 0]  # close x face
lett_face[4] = [2, 0]  # close z face
lett_face[5] = [1, 1]  # far y face

fcol = np.full(n_voxels.shape, boxcol)
ecol = np.full(n_voxels.shape,  edgecol)
filled = np.ones(n_voxels.shape)

# upscale the above voxel image, leaving gaps
filled_2 = explode(filled)
fcolors_2 = explode(fcol)
ecolors_2 = explode(ecol)

# Shrink the gaps
corn = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
ccorn = 0.05  # close corner
fcorn = 1.0 - ccorn
corn[0, 0::2, :, :] += ccorn
corn[1, :, 0::2, :] += ccorn
corn[2, :, :, 0::2] += ccorn
corn[0, 1::2, :, :] += fcorn
corn[1, :, 1::2, :] += fcorn
corn[2, :, :, 1::2] += fcorn
#corn[...] += 2.5
#corn[...] *= 1000.0
# print(corn[:,:2,:2,:2])
#print(corn.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.axis("off")

# Plot the voxels
x, y, z = corn
ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)

# Plot the letter square faces
jj = 0
myo = 0.005  # my face offset
for j in [x for x in letters if x is not None]:

    locf = np.empty((j.shape[0], 3, 2, 2))  # local face

    ji = 0
    for i in j:
        i = (i + 1) * 2  # skip empty voxels
        loc = corn[:, i[0]:i[0]+2, i[1]:i[1]+2, i[2]:i[2]+2]  # local corners
        locf[ji] = voxel_face(loc, lett_face[jj, 0], lett_face[jj, 1], myo)
        ax.plot_surface(locf[ji, 0], locf[ji, 1], locf[ji, 2], 
                        color=faceCols[jj], shade=False)
        # ffe749
        # ffe500
        ji += 1

    jj += 1

# plt.show()
#'''
ssp =  8 #small step
bsp = 40 #big step
sps = 15 #short pause
#Views:        PY,  P,   Y,  T,   H,   O,    N,   spiiiiiiiiiiiiin,  PY
view_elev = [  5,   0,   0,  90,   0, -90,   0,   5,  10,  11,  10,   5]
view_azim = [-60, -90,   0,  90, 180, 180,  90,  60,  30,   0, -30, -60]
view_step = [ssp, bsp, bsp, bsp, bsp, bsp, ssp, ssp, ssp, ssp, ssp]
view_paus = [ 40, sps, sps, sps, sps, sps, sps,   0,   0,   0,   0]

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=25, metadata=metadata)

with writer.saving(fig, "voxelRotate.mp4", 200):


    for i in range(1,len(view_elev)):

        de = (view_elev[i] - view_elev[i-1])
        da = (view_azim[i] - view_azim[i-1])

        if abs(da) >= 180 : #unecessary in this config
            da -= copysign(360, da)
        if abs(de) >= 180 :
            de -= copysign(360, de)

        steps = view_step[i-1]
        da = da / steps
        de = de / steps

        for j in range(view_paus[i-1]): #Pause on direct view of a letter
            ax.view_init(view_elev[i-1], view_azim[i-1])
            plt.draw()
            writer.grab_frame()
        for j in range(steps): #Rotate to next letter
            ax.view_init(view_elev[i-1] + j*de,
                         view_azim[i-1] + j*da)
            plt.draw()
            writer.grab_frame()
#'''