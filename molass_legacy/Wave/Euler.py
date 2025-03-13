"""
Beautiful equation - Euler's identity

The code prints the mathematics 'beautiful equation' or 'Euler's identity' result.
The formula is:

        e**πi + 1 = 0

where:
e - Euler's number, 2.718...
π - pi, 3.14159...
i - imaginary number, equal to √-1 (i**2 = -1)

The formula is called 'beautiful' because it uses the mathematics most widely important constants:
e - the base of natural logarithms
π - the ratio of circumference to diameter
i - imaginary unit of complex numbers
0 - additive identity (n + 0 = n)
1 - multiplicative identity (n * 1 = n)

"""

import math
print (f"{(math.e**(math.pi*1j)) + 1:.12f}")
