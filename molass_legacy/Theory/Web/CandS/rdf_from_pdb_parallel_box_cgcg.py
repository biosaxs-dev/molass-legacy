#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Generate RDF from PDB
# Copyright (C) 2011 Stas Bevc, stas.bevc@cmm.ki.si
# www.sicmm.org

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>

from multiprocessing import Pool, cpu_count
from sys import argv
from os import listdir
from math import sqrt, ceil, floor, pow, pi
import time


# generate radial distribution function from
# particle positions stored in PDB files
def makeRDF(pdbdir, pdbs, side, numbins, cm, numAT=0):
    
    maxbin = numbins # number of bins
    sideh = side[1]/2.0 # take y dir
    sidehx = side[0]/2.0
    sidehy = side[1]/2.0
    sidehz = side[2]/2.0
    dr = float(sideh/maxbin) # bin width
    hist = [0]*(maxbin+1) # initialize list
    count = 0
    cnt = 0
    
    # list of pdbs is now an input
    #pdbs = listdir(pdbdir) # list of files
    nstep = len(pdbs)
    
    cutsq = sideh*sideh
    
    mO = 1
    mH = 0.063002362588597
    mMol = 2*mH + mO
    
    # loop over pdb files
    for pdb in pdbs:
        
        #if not pdb.endswith(".pdb"):
        #    nstep -= 1
        #    continue # skip other files
            
        count += 1
        print "Reading file ... "+pdb+" ("+ str(count) +"/"+ str(len(pdbs)) +")"
        
        # read atom coordinates from PDB
        atoms = []
        cmm = [0,0,0] # center of mass of molecule
        atc = 1 # atom counter
        lines = open(pdbdir+"/"+pdb)
        for line in lines:
            if line[0:4] != "ATOM":
                continue # skip other lines
            coords = map(float, (line[31:54]).split())
            label = line[13:16] # atom label
            atype = line[77:78] # atom type
            
            if label == "NA " or label == "CL ":
                continue # skip other atoms
            
            
            if cm == True: # calculate center of mass
                if label == "OH ": # O
                    for i in range(3):
                        cmm[i] += mO * coords[i]
                if label == "1HH" or label == "2HH": # H1, H2
                    for i in range(3):
                        cmm[i] += mH * coords[i]
                if atc < numAT:
                    atc += 1
                else:
                    atc = 1
                    for i in range(3):
                        cmm[i] /= mMol
                    
                    # fold coordinates
                    for i in range(3):
                        tmp = floor(cmm[i] * (1.0/side[i]))
                        cmm[i] -= tmp * side[i]
                    
                    atoms.append((cmm[0],cmm[1],cmm[2]))
                    cmm = [0,0,0]
            else: # no cm calculation
                atoms.append((coords[0], coords[1], coords[2]))
        
        # loop over particle pairs
        npart = len(atoms)
        print " looping over particle pairs (" +str(npart)+ "^2) ... "
        for i in range(npart):
            
            xi = (atoms[i])[0]
            yi = (atoms[i])[1]
            zi = (atoms[i])[2]
            
            for j in range(i+1, npart):
                xx = xi - (atoms[j])[0]
                yy = yi - (atoms[j])[1]
                zz = zi - (atoms[j])[2]
                
                # minimum image
                if (xx < -sidehx):   xx = xx + side[0]
                elif (xx > sidehx):    xx = xx - side[0]
                if (yy < -sidehy):   yy = yy + side[1]
                elif (yy > sidehy):    yy = yy - side[1]
                if (zz < -sidehz):   zz = zz + side[2]
                elif (zz > sidehz):    zz = zz - side[2]
                
                # square distance between i and j
                rd  = xx * xx + yy * yy + zz * zz
                
                if rd < cutsq:
                    rij = sqrt(rd)
                    bin = int(ceil(rij/dr)) # determine in which bin the distance falls
                    if (bin <= maxbin):
                        hist[bin] += 1
                cnt += 1
    
    return (hist, cnt)
    
    
def normalize(hist, side, cnt):
    
    print "Normalizing ... "
    rdf = {}
    maxbin = (len(hist)-1)
    dr = float((side[1]/2.0)/maxbin) # bin width
    vol = side[0] * side[1] * side[2]  #pow(side, 3.0)
    
    for i in range(1, maxbin+1):
        rrr = (i - 0.5) * dr
        bin_vol = (4.0/3.0) * pi * (pow(rrr+(dr/2.0),3.0) - pow(rrr-(dr/2.0), 3.0))
        val = ((hist[i] * vol) / (cnt * bin_vol))
        rdf.update({rrr:val})
        
    #print "Normalizing ... "
    #rdf = {}
    #maxbin = (len(hist)-1)
    #dr = (side/2.0)/maxbin # bin width
    #phi = npart/pow(side, 3.0) # number density (N/V)
    #norm = 2.0 * pi * dr * phi * nstep * npart
    
    #print npart, nstep
    
    #for i in range(1, maxbin+1):
        #rrr = (i - 0.5) * dr
        #val = hist[i]/ norm / ((rrr * rrr) + (dr * dr) / 12.0)
        #rdf.update({rrr:val})
    
    return rdf




#-------------------------------------------------------------------#

if __name__ == '__main__':
    cpucount = cpu_count()
    pool = Pool(processes=cpucount)  # start worker processes

    # RDF setup
    pdbdir = argv[1] # directory with PDB files (eg. "./pdb-ex-10k/")
    boxsize = (51.16866708780796, 12.79216677195199, 12.79216677195199)
    numbins = 300 # number of bins
    outfile = pdbdir + "rdf.out"
    
    # calculate RDF from center of mass of molecule
    numAT = 3 # number of atoms in molecule
    cm = False # default to false
    #if len(argv) == 3:
    #    cm = argv[2]
    
    # prepare lists of pdb files
    files = listdir(pdbdir) # list of files
    nfiles = 0 # number of total pdb files
    pdbs = [] # list of pdb files
    for f in files:
        if not f.endswith(".pdb"):
            continue # skip other files
        nfiles += 1
        pdbs.append(f)
    
    if cpucount > nfiles:
        cpucount = nfiles
    
    files_oncpu = int(floor(nfiles / cpucount))
    dist_pdbs = [] # pdbs to distribute
    p = 0
    for i in range (cpucount):
        tmp = []
        for j in range(files_oncpu):
            tmp.append(pdbs[p])
            p += 1
        dist_pdbs.append(tmp)
    
    # distribute remaining files
    rem = nfiles - (files_oncpu * cpucount)
    if (rem > 0):
        for i in range(rem):
            dist_pdbs[i].append(pdbs[(files_oncpu * cpucount)+i])
    
    # distribute lists to pools
    results = [ pool.apply_async(makeRDF, [pdbdir,dist_pdbs[i],boxsize,numbins,cm,numAT])
                for i in range(len(dist_pdbs)) ]
    
    # sum up results
    hist = [0]*(numbins+1) # initialize list
    cnt = 0;
    for i in range(len(dist_pdbs)):
        result = results[i].get()
        cnt += result[1]
        for j in range(numbins+1):
            hist[j] += result[0][j]
    
    #print hist

    rdf = normalize(hist, boxsize, cnt)
    
    print "Writing output file ... " +outfile+ " (" + time.strftime("%c") +")"
    outfile = open(outfile, "w")
    for r in sorted(rdf.iterkeys()): # sort before writing into file
        outfile.write("%15.8g %15.8g\n"%(r, rdf[r]))
    outfile.close()



