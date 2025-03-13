#!/usr/bin/env python

# Calculate g(r) from a trajectory PDB file
# Chris Cioce, 2012, ccioce@mail.usf.edu
# Original script by Stas Bevc, 2011, stas.bevc@cmm.ki.si, www.sicmm.org

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

"""
BACKGROUND:
This script will calculate the 3D radial distribution function from a specified trajectory PDB file.
Trajectory PDB files are individual PDB files strung together, separated by the keyword "ENDMDL".
It is important for viewers, such as VMD (http://www.ks.uiuc.edu/Research/vmd/), to ensure that 
each of the configurations (idividual frames) has the same number of atoms. This can be problematic 
for viewing trajectories of Monte Carlo simulations. There are, however, ways of tricking the viewers 
to play the files. A perl script which does this can be downloaded from the MPMC/tools/ directory 
(called vmdfixn.pl) at http://code.google.com/p/mpmc/source/browse/#svn%2Ftrunk%2Ftools.

WHAT YOU NEED TO MODIFY IN THIS SCRIPT:
This script assumes that there are center of mass sites for every molecule in the trajectory PDB file.
For example, if N2 is the species under consideration, there assumes that halfway between the nitrogens
resides a site with definitive coordinates, let's call it "N2G". For molecules such as CO2 and CH4,
this is not an issue, as the carbon atoms are the center of masses of the molecules. If you do not 
have these center of mass sites for your molecule(s) of interest, consider downloading the original 
python script (http://www.sicmm.org/~stas/rdf-script/) from which mine is a modified version of.
This original script has a flag (cm=True) which will auto-magically compute the COM sites for your
molecules.

This only one line which needs to be changed for you to successfully run this script, is the line

if line[12:15] != "N2G":                   # XXX Identifying label for center of gravity (COM) site

This line (line 67) is clearly marked with XXX, which highlights in editors such as vim.
"""

import sys
from os import listdir, getcwd
from math import sqrt, ceil, floor, pow, pi

# Compute g(r) from a trajectory PDB file
def calcRDF(trajfile, pdbdir, side, n_bins):

    print "Calculating g(r)..."
    traj = open(pdbdir+"/"+trajfile)

    maxbin = n_bins                                                                     # Number of requested bins
    box_half = side * 0.5
    dr = float(box_half/maxbin)                                                         # Bin width
    hist = [0]*(maxbin+1)
    rdf = {}
    config = 1                                                                          # Configuration (step) counter
    part_total = 0
    
    # Extract COM sites from trajectory PDB
    atoms = []
    for line in traj:                                                                   # Read entire trajectory file

        if line[12:15] != "N2G":                                                        # XXX Identifying label for center of gravity (COM) site

            if line[0:6] == "ENDMDL":                                                   # We've reached the end of a configuration
                # loop over particle pairs
                npart = len(atoms)
                part_total += npart
                for i in range(npart):
                        
                    xi = (atoms[i])[0]
                    yi = (atoms[i])[1]
                    zi = (atoms[i])[2]
                        
                    for j in range(i+1, npart):
                        xx = xi - (atoms[j])[0]
                        yy = yi - (atoms[j])[1]
                        zz = zi - (atoms[j])[2]
                            
                        # minimum image
                        if (xx < -box_half):   xx = xx + side
                        if (xx >  box_half):   xx = xx - side
                        if (yy < -box_half):   yy = yy + side
                        if (yy >  box_half):   yy = yy - side
                        if (zz < -box_half):   zz = zz + side
                        if (zz >  box_half):   zz = zz - side
                          
                        # distance between i and j
                        rd  = xx * xx + yy * yy + zz * zz
                        rij = sqrt(rd)
                           
                        bin = int(ceil(rij/dr)) # determine in which bin the distance falls
                        if (bin <= maxbin):
                            hist[bin] += 1

                # Reset Parameters
                atoms = []
                config += 1                                                                     # Update config counter

            continue                                                                            # We're not interested in non-COM atoms
        coords = map(float, (line[31:54]).split())                                              # However, if we recognize a COM site, map the cartesian coords
        atoms.append((coords[0], coords[1], coords[2]))

    traj.close()

    # Normalize
    print "Normalizing ... "
    nconfigs = config - 1
    avg_part = part_total / float(nconfigs)
    rho = avg_part/pow(side, 3.0)                                                               # Density
    norm = 2.0 * pi * dr * rho * nconfigs * avg_part

    print "NORM: %f\tDR: %f\tRHO: %f\t NCONFIGS: %d\tAVG_PART: %f" % (norm,dr,rho,nconfigs,avg_part)

    for i in range(1, maxbin+1):
        rrr = (i - 0.5) * dr
        val = hist[i]/ norm / ((rrr * rrr) + (dr * dr) / 12.0)
        rdf.update({rrr:val})
    
    return rdf

#-------------------------------------------------------------------#

# Write to file
pdbsdir = getcwd() # directory with PDB files
rdfout  = "./RDF.dat"

# Check Arguments
if len(sys.argv) == 4:
    trajfile      = sys.argv[1]
    boxlength     = float(sys.argv[2])
    n_bins        = int(sys.argv[3])
else:
    print "Insufficent number of arguments. USAGE: $ python RDF.py [traj_file.pdb] [box_length (A)] [# of bins]"
    sys.exit(1)

rdf = calcRDF(trajfile, pdbsdir, boxlength, n_bins)

print "Writing output file ... " + rdfout
rdfout = open(rdfout, "w")
for r in sorted(rdf.iterkeys()):                                                # Sort
    rdfout.write("%15.8g %15.8g\n"%(r, rdf[r]))
rdfout.close()
