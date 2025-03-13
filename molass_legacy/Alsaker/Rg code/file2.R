###############################################################################
# Example of a single replicate.
# Set working directory where the example data are stored, 
# either via the pull-down menu: File->Change dir...
# or via setwd("Drive:/Some Directory/FirstSubdirectory/.../LastSubdirectory")
# In the file pathname, subdirectories MUST be separated by a forward slash (/) not a  
# backward slash(/), regardless of operating system.
# This example is a standard text file (.dat, .txt, etc.) without a header: data must be space or tab-delimited.
# Run the following code by highlighting and pressing Ctrl+R:
#
# Read the data by navigating to the file:  
data = read.table(file.choose(),header=FALSE)
# Alternatively, read the data by specifying the file name:
data = read.table("oval_01C_S008_0_01.dat", header = FALSE)
# Once the data are read, by either of the methods above, run the estimation code:

estimate_Rg(data, 1, 5) # single replicate with initial angle 5th in the data set (deleting first 4), as specified by user
estimate_Rg(data, 1)    # single replicate with initial angle selected automatically (default if no initial angle is provided)


######################################################################################
# Variations on data input:
# For a standard text file (.dat, .txt, etc.) with a one-line header of variable names, 
# replace header=FALSE with header=TRUE in the above read.table statements:
# data = read.table("example.txt", header = TRUE)
# or
# data = read.table(file.choose(),header=TRUE)
#
# For a CSV file (.csv, containing comma-delimited data) without a one-line header, use
# data = read.csv("example.csv",header=FALSE)
# For a CSV file with a one-line header, use
# data = read.csv("example.csv",header=TRUE)
######################################################################################
