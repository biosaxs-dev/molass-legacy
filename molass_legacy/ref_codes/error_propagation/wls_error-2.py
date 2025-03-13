import numpy    as np
import matplotlib.pyplot as plt
import statsmodels.api      as sm

a = 2.5
b = 1.1

x = np.linspace( 0, 5, 10 )

y = a*x + b

np.random.seed( 123 )
obs = y + 0.5 * np.random.randn(10)
err = 0.3 * np.random.randn(10)
w = 1/err**2

X = sm.add_constant(x)
result = sm.WLS(obs, X, weights=w).fit()
# result = sm.OLS(obs, X).fit()
b_, a_ = result.params

print( 'result.cov_params()=',  np.diag( result.cov_params() ) )

y_ = a_*x + b_

plt.plot( x, y, color='green', label='True' )
plt.plot( x, obs, 'o', color='blue', label='Observed' )
plt.errorbar( x, obs, yerr=err, color='blue', fmt='none' )
plt.plot( x, y_, color='red', label='Fitted' )
plt.legend( loc="upper left" )
plt.show()
