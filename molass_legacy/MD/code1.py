"""
https://arxiv.org/pdf/2103.16944.pdf

A CONCISE INTRODUCTION TO MOLECULAR DYNAMICS
    SIMULATION: THEORY AND PROGRAMMING

Ashkan Shekaari, Mahmoud Jafari
April 1, 2021

code1.py
"""
import numpy as np
from math import *
import random
__d__ = 3
NS = 1000
N = 100
dt = 1.e-4
T_0 = 300.
sig = 1.
eps = 1.
r_ctf = 2.5* sig
u_at_ctf = 4.* eps *(( sig / r_ctf ) **12 - ( sig/ r_ctf ) **6)
du_at_ctf = 24.* eps *( -2.* sig **12/ r_ctf **13 \
                    + sig **6/ r_ctf **7)
bs = 10.* sig
vol = bs **3
rho = N/ vol
ign = 20
#-------------------------------------------------
# Initiate positions, velocities, & accelerations
pos = np.zeros([__d__, N])
vel = np.zeros([__d__, N])
acc = np.zeros([__d__, N])
for j in range (N):
    for i in range (__d__):
        pos[i,j] = random.uniform(0, bs)
        vel[i,j] = random.uniform(0, 1)
#--------------------------------------
# Scale positions to the box size .
pos = pos/bs
#----------------------------
# Translate positions to the
# center -of - mass ( com) reference frame .
com = np . zeros ([ __d__ ])
for i in range ( __d__ ):
    for j in range (N):
        com [i] += pos[i , j]
        com [i] = com [i ]/ N
for i in range ( __d__ ):
    pos [i , :] -= com [i]
# - - - - - - - - -- ---- --- ----
P_S = 0.
k_S = 0.
p_S = 0.
T_S = 0.
f_tpz = open ('tpz. out', 'w')
f_kpe = open ('kpe. out', 'w')
f_xyz = open ('pos. xyz', 'w')
R = np . zeros ([ __d__ , N , N ])
r = np . zeros ([ __d__ , N , N ])
# - - - - - - - - -- ---- --- ---- --- --
def sign (a , b):
    """
    The sign function
    """
    if b > 0:
        return a
    elif b < 0:
        return -a
    elif b == 0:
        return a
# - - - - - - - - -- ---- --- ---- --- --- ---- --- --
# The time evolution loop ( main loop )
for step in range (1 , NS + 1) :
    # - - - - - - - - --- --- ---- --- --- ---- -
    # Periodic boundary conditions
    pos [ np . where ( pos > 0.5) ] -= 1
    pos [ np . where ( pos < -0.5) ] += 1
    # - - - - - - - - --- --- ---- --- --- ---- -
    # Loop for computing forces
    acc = np . zeros ([ __d__ , N ])
    pot = np . zeros ([ N ])
    vrl = 0.
    for i in range ( N):
        for j in range (i + 1 , N) :
            if j != i:
                r2 = np . zeros ([N , N ])
                for k in range ( __d__ ):
                    R[k , i , j] = pos [k , i] - pos[k , j]
                    if abs (R [k , i , j ]) > 0.5:
                        R[k , i , j ] -= sign (1. , R[k , i , j ])
                    r[k , i , j] = bs *R[k , i , j]
                    r2 [i , j] += r[k , i , j ]* r[k , i , j ]
            if r2 [i , j] < r_ctf * r_ctf :
                r1 = sqrt ( r2 [i , j ])
                ri2 = 1./ r2 [i , j ]
                ri6 = ri2 * ri2 * ri2
                ri12 = ri6 * ri6
                sig6 = sig **6
                sig12 = sig **12
                u = 4.* eps *( sig12 * ri12 - sig6 * ri6 ) \
                        - u_at_ctf - r1 * du_at_ctf
                du = 24.* eps * ri2 *(2.* sig12 * ri12 - sig6 * ri6) \
                        + du_at_ctf* sqrt ( ri2 )
                pot [j ] += u
                vrl -= du * r2 [i , j]
                for k in range ( __d__ ):
                    acc [k , i ] += du *R[k , i , j ]
                    acc [k , j ] -= du *R[k , i , j ]
    vrl = - vrl / __d__
    # - - - - - - - - --- --- ----
    # Update positions
    pos += dt * vel + 0.5* acc * dt * dt
    # - - - - - - - - --- --- ---- --- --- ----
    # Compute temperature
    kin = np . zeros ([ N ])
    v2 = np . zeros ([ N ])
    for j in range ( N):
        for i in range ( __d__ ) :
            v2[j ] += vel[i, j]*vel[i, j]* bs * bs
        kin[ j] = 0.5* v2 [j]
    k_AVG = sum( kin )/N
    T_i = 2.* k_AVG / __d__
    B = sqrt ( T_0 / T_i )
    # - - - - - - - - --- --- ---- --- --- ---- --- -
    # Rescale & update the velocities
    # according to velocity Verlet algorithm
    vel = B* vel + 0.5* dt * acc
    vel += 0.5* dt * acc
    # - - - - - - - - --- --- ---- ---
    # Compute temperature
    kin = np . zeros ([ N ])
    v2 = np . zeros ([ N ])
    for j in range ( N):
        for i in range ( __d__ ) :
            v2 [j ] += vel [i , j ]* vel [i , j ]* bs * bs
        kin[ j] = 0.5* v2 [j]
    k_AVG = sum( kin )/N
    T_i = 2.* k_AVG / __d__
    B = sqrt ( T_0 / T_i )
    # - - - - - - - - --- --- ----
    p_AVG = sum( pot )/N
    etot_AVG = k_AVG + p_AVG
    P = rho* T_i + vrl / vol
    Z = P * vol /( N * T_i )
    f_tpz . write ("%e %e % e %e\n " \
    %( step *dt , T_i , P , Z))
    f_kpe . write ("%e %e % e %e\n " \
    %( step *dt , k_AVG , p_AVG , etot_AVG))
    # - - - - - - - - --- --- ---- --- --- ---- --- ---
    # Write position components
    f_xyz . write ("%d \n\n " %( step ))
    for i in range ( N):
        f_xyz . write ("%e % e %e\n " %( pos [0 , i ]* bs , \
        pos [1 , i ]* bs , pos [2 , i ]* bs ))
    # - - - - - - - - --- --- ---- --- --- ---- --- -
    if step > ign :
        P_S += P
        k_S += k_AVG
        p_S += p_AVG
        T_S += T_i
    # - - - - - - - - -- ---- --- ---- ---
    if step %( NS *0.1) == 0: # To inform that the code is being run
        print (">>>>> Iteration # %d out of %d " %( step , NS ) )
print ('\n >>>>> Statistical Averages <<<<<\n')
print ('Temperature = %f K' %( T_S /( NS - ign )) )
print ('Kinetic energy = %f eps' %( k_S /( NS - ign )))
print ('Potential energy = %f eps' %( p_S /( NS - ign )))
print ('Total energy = %f eps' %(( k_S + p_S ) /( NS - ign )) )
print ('Pressure = %f eps /( sig ^3) \n' %( P_S /( NS - ign )))
