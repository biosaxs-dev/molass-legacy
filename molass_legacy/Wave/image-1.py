import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100).reshape(-1,1)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,7))

data = x+y
print(x.shape, y.shape, data.shape)
ax1.imshow(data, animated=True)

fig.tight_layout()
plt.show()
