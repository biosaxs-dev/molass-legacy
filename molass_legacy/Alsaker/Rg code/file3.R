######################################################################################
# Example of multiple replicates (up to ten in this example).
# Set working directory where the example data are stored, 
# either via the pull-down menu: File->Change dir...
# or via setwd("Drive:/Some Directory/FirstSubdirectory/.../LastSubdirectory")
# In the file pathname, subdirectories MUST be separated by a forward slash (/) not a  
# backward slash(/), regardless of operating system.
# In this example, the data are stored in ten standard text files
# (.dat, .txt, etc.), each without a header: data must be space or tab-delimited.
# See file2.R for alternate file formats.
# Run the following code by highlighting and pressing Ctrl+R:
#
data1 = read.table("myo2_07D_S215_0_01.dat", header = FALSE)
data2 = read.table("myo2_07D_S215_0_02.dat", header = FALSE)
data3 = read.table("myo2_07D_S215_0_03.dat", header = FALSE)
data4 = read.table("myo2_07D_S215_0_04.dat", header = FALSE)
data5 = read.table("myo2_07D_S215_0_05.dat", header = FALSE)
data6 = read.table("myo2_07D_S215_0_06.dat", header = FALSE)
data7 = read.table("myo2_07D_S215_0_07.dat", header = FALSE)
data8 = read.table("myo2_07D_S215_0_08.dat", header = FALSE)
data9 = read.table("myo2_07D_S215_0_09.dat", header = FALSE)
data10= read.table("myo2_07D_S215_0_10.dat", header = FALSE)
#
# For illustration, look at one replicate, three replicates, and ten replicates.
# First combine the data into one big ten-replicate matrix.
# Keep angle and intensity from replicate 1 (columns 1 and 2 but not 3),
# intensity from replicate 2 (column 2 only),
# intensity from replicate 3 (column 2 only),...,
# intensity from replicate 10 (column 2 only).
#
combined_data = cbind(data1[1:400,-3],data2[1:400,2],data3[1:400,2],
data4[1:400,2],data5[1:400,2],data6[1:400,2],
data7[1:400,2],data8[1:400,2],data9[1:400,2],
data10[1:400,2])
#
# Run the estimation code with one replicate
# (only the first two columns of the combined data), with no points deleted:
estimate_Rg(combined_data[,1:2], 1, 1)
# Run the estimation code with three replicates
# (only the first four columns of the combined data), with no points deleted:
estimate_Rg(combined_data[,1:4], 3, rep(1,3))
# Run the estimation code with all ten replicates
# (all eleven columns of the combined data), with no points deleted:
estimate_Rg(combined_data, 10, rep(1,10))
#
# The third argument can be altered to allow for a different number of deleted points for each replicate.  
# For example, if you want to eliminate the first three points of the 
# fourth replicate while deleting no points from the other nine replicates, you would use the following code:
#
estimate_Rg(combined_data,10,c(1,1,1,4,1,1,1,1,1,1))
#
# The third argument can be eliminated, in which case the program uses an 
# outlier detection algorithm to determine one, common initial point for all of 
# of the replicate curves. 
#
# Automatic selection of inital point with 3 replicates:
estimate_Rg(combined_data[,1:4],3) 
# Automatic selection of inital point with 10 replicates: 
estimate_Rg(combined_data, 10)     
######################################################################################


