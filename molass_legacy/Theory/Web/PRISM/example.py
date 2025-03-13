import pyPRISM
import matplotlib.pyplot as plt

# The system holds all information needed to set up a PRISM problem. We
# instantiate the system by specifying the site types and thermal energy
# level (kT, coarse-grained temperature) of the system.
sys = pyPRISM.System(['particle','polymer'],kT=1.0)

# We must discretize Real and Fourier space
sys.domain = pyPRISM.Domain(dr=0.01,length=4096)

# The composition of the system is desribed via number densities
sys.density['polymer']  = 0.75
sys.density['particle'] = 6e-6

# The diameter of each site is specified (in reduced units)
sys.diameter['polymer']  = 1.0
sys.diameter['particle'] = 5.0

# The molecular structure is described via intra-molecular correlation
# functions (i.e. omegas)
sys.omega['polymer','polymer']   = pyPRISM.omega.FreelyJointedChain(length=100,l=4.0/3.0)
sys.omega['polymer','particle']  = pyPRISM.omega.NoIntra()
sys.omega['particle','particle'] = pyPRISM.omega.SingleSite()

# The site-site interactions are specified via classes which are lazily
# evaluated during the PRISM-object creation
sys.potential['polymer','polymer']   = pyPRISM.potential.HardSphere(sigma=1.0)
sys.potential['polymer','particle']  = pyPRISM.potential.Exponential(sigma=3.0,alpha=0.5,epsilon=1.0)
sys.potential['particle','particle'] = pyPRISM.potential.HardSphere(sigma=5.0)

# Closure approximations are also specified via classes
sys.closure['polymer','polymer']   = pyPRISM.closure.PercusYevick()
sys.closure['polymer','particle']  = pyPRISM.closure.PercusYevick()
sys.closure['particle','particle'] = pyPRISM.closure.HyperNettedChain()

# Calling the .solve() method of the system object attempts to numerically
# solv the PRISM equation and, if successful, it returns a PRISM object
# containing all of the solved correlation functions.
PRISM = sys.solve()

# Calculate the pair-correlation functions.
rdf = pyPRISM.calculate.pair_correlation(PRISM)

# Plot the results using matplotlib
plt.plot(sys.domain.r,rdf['polymer','polymer'],color='gold',lw=1.25,ls='-')
plt.plot(sys.domain.r,rdf['polymer','particle'],color='red',lw=1.25,ls='--')
plt.plot(sys.domain.r,rdf['particle','particle'],color='blue',lw=1.25,ls=':')
plt.ylabel('pair correlation')
plt.xlabel('separation distance')
plt.show()
