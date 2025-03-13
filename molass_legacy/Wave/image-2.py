import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

canvas = np.zeros((200, 200))

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,7))

n = 120
x = np.linspace(0, np.pi*12, n)

ims = []
for i in range(60):
    t = np.pi*i/3
    beam = 1 + np.sin(x - t)
    canvas[100,0:n] = beam
    im = ax1.imshow(canvas, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
fig.tight_layout()
plt.show()
