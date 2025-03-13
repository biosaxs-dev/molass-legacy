import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

N = 200
n = N//2
r = np.arange(N)
rr = r**2
i = np.arange(N)
ii, jj = np.meshgrid(i, i)
labels = np.searchsorted(rr, (ii - n)**2 + (jj-n)**2)
print(labels.shape)
Lmax = np.max(labels)
print(labels[0:3,0:3])

canvas = np.zeros((N, N))

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,7))

# ax2.imshow(labels)

x = np.linspace(0, np.pi*12, N + Lmax)

ims = []
for i in range(60):
    t = np.pi*i/3
    beam = 1 + np.sin(x[0:n] - t)
    canvas[100,0:n] = beam
    im = ax1.imshow(canvas, animated=True)
    ims.append([im])
    scattered = 1 + np.sin(x[n:] - t)
    im = ax2.imshow(scattered[labels], animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
fig.tight_layout()
plt.show()
