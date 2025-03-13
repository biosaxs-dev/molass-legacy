import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.animation as animation

x = np.linspace(0, 10*np.pi, 100)
y = np.sin(x)
fig, (ax1, ax2) = plt.subplots(nrows=2)
line, = ax1.plot(x, y)

ax2.set_ylim(0,1)
z = np.ones(len(x))*0.5
artists = ax2.scatter(x, z, vmin=-1, vmax=1, c=y, cmap=cm.plasma)

def init():
  line.set_data([],[])
  artists.set_array(y)
  return [line, artists]

num_frames = 40

def animate(i):
  y = np.sin(x-10*np.pi*i/num_frames)
  line.set_data(x,y)
  artists.set_array(y)
  return [line, artists]

anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=num_frames, interval=100, blit=True)
# anim.save('beam_wave.mp4', writer="ffmpeg")
plt.show()
