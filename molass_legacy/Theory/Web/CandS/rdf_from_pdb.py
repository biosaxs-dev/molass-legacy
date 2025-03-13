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


from os import listdir
from math import sqrt, ceil, floor, pow, pi


# generate radial distribution function from
# particle positions stored in PDB files
def makeRDF(pdbdir, side, numbins, cm=False, numAT=0):
    
    maxbin = numbins # number of bins
    sideh = side/2.0
    dr = float(sideh/maxbin) # bin width
    hist = [0]*(maxbin+1) # initialize list
    rdf = {}
    count = 0
    pdbs = listdir(pdbdir) # list of files
    nstep = len(pdbs)
    
    print "Directory "+pdbdir+" has "+str(len(pdbs))+" files."
    
    # loop over pdb files
    for pdb in pdbs:
        if not pdb.endswith(".pdb"):
            nstep -= 1
            continue # skip other files
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
            
            if cm == True: # calculate center of mass
                # we assume masses of all particles are 1.0
                cmm[0] += coords[0]
                cmm[1] += coords[1]
                cmm[2] += coords[2]
                if atc < numAT:
                    atc += 1
                else:
                    atc = 1
                    cmm[0] /= numAT
                    cmm[1] /= numAT
                    cmm[2] /= numAT
                    
                    # fold coordinates
                    for i in range(3):
                        tmp = floor(cmm[i] * (1.0/side))
                        cmm[i] -= tmp * side
                    
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
                if (xx < -sideh):   xx = xx + side
                if (xx > sideh):    xx = xx - side
                if (yy < -sideh):   yy = yy + side
                if (yy > sideh):    yy = yy - side
                if (zz < -sideh):   zz = zz + side
                if (zz > sideh):    zz = zz - side
                
                # distance between i and j
                rd  = xx * xx + yy * yy + zz * zz
                rij = sqrt(rd)
                
                bin = int(ceil(rij/dr)) # determine in which bin the distance falls
                if (bin <= maxbin):
                    hist[bin] += 1
    
    # normalize
    print "Normalizing ... "
    phi = npart/pow(side, 3.0) # number density (N*V)
    norm = 2.0 * pi * dr * phi * nstep * npart
    
    for i in range(1, maxbin+1):
        rrr = (i - 0.5) * dr
        val = hist[i]/ norm / ((rrr * rrr) + (dr * dr) / 12.0)
        rdf.update({rrr:val})
    
    return rdf

#-------------------------------------------------------------------#

# write RDF into file
boxsize = 36.845
numbins = 384 # number of bins
cm = True # calculate RDF from center of mass of molecule
numAT = 4 # number of atoms in molecule
pdbsdir = "./pdbs/" # directory with PDB files
outfile = "./rdf.out"

rdf = makeRDF(pdbsdir, boxsize, numbins, cm, numAT)
print "Writing output file ... " +outfile
outfile = open(outfile, "w")
for r in sorted(rdf.iterkeys()): # sort before writing into file
    outfile.write("%15.8g %15.8g\n"%(r, rdf[r]))
outfile.close()



