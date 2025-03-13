import numpy                as np
import matplotlib.pyplot    as plt

y = 5 + 0.2 * np.random.randn( 10000 )

stdev = np.std( y )
print( 'std(y)=', stdev )

yn_stdev = np.std( [ np.average( y[i*10:(i+1)*10] ) for i in range(100) ] )

print( 'yn_stdev=', yn_stdev )
print( 'stdev/sqrt(10)=', stdev/np.sqrt(10)  )
