"""

    1) python create_icon_image.py
    2) convert at https://converticon.com/
    3) as administrator
       rcedit.exe path_dir\serial_analyzer.exe --set-icon path_dir\serial_analyzer.ico

       F:\Tools\rcedit\rcedit-master\Default\rcedit.exe

"""
import numpy                as np
import matplotlib.pyplot    as plt
from scipy                  import misc
from scipy.stats            import norm

image = np.zeros( ( 64, 64, 4 )  )
image[:, :] =  [ 255, 255, 255, 255 ]

x   = np.arange( 64, dtype=int )
y   = norm.pdf( x, 32, 6 )
y_  = 60 * y / y[32]

for i in x:
    j = 62 - int(y_[i])
    js_ = slice( max(0, j-1), min(64, j+2) )
    is_ = slice( max(0, i-1), min(64, i+2) )
    image[ js_, is_ ] = [ 0, 0, 255, 255 ]

image[ 62:64, : ] = [ 255, 0, 0, 255 ]
image[ 0, : ] = [ 127, 127, 127, 255 ]
image[ :, 0 ] = [ 127, 127, 127, 255 ]
image[ :, 63 ] = [ 127, 127, 127, 255 ]

misc.imsave( 'icon.png', image )

plt.imshow( image )
plt.tight_layout()
plt.show()
