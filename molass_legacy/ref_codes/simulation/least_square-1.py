import numpy                as np
import matplotlib.pyplot    as plt

N   = 10
a   = 5
b   = 3
s   = 0.2
x   = np.linspace( 0, 1, N )
np.random.seed( 1234 )
e   = np.random.normal( 0, s, N )
y   = a * x + b + e

yt  = a * x + b

plt.plot( x, y, 'o' )
plt.plot( x, yt, ':r' )
plt.show()
