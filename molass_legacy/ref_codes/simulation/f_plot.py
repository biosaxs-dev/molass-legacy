import numpy    as np
import matplotlib.pyplot    as plt
import seaborn              as sns
from Intensity              import f

x = np.linspace( 0, 10, 100 )

plt.title( 'f(sR) = 3 * ( np.sin( sR ) - sR * np.cos( sR ) ) / ( sR )**3' )
plt.ylabel( 'Intensity' )
plt.xlabel( 'sR' )
plt.plot( f(x) )
plt.show()
