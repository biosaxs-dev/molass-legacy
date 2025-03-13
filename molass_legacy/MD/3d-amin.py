"""
    from
    https://start-python.hateblo.jp/entry/2020/01/18/090000
"""
import random
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Qテーブルの初期化
q_table_1 = np.zeros((6, 6))
q_table_2 = np.zeros((6, 6))

# ３Dグラフの準備
fig = plt.figure()

x = [1, 2, 3, 4, 5, 6]
y = [1, 2, 3, 4, 5, 6]
X, Y = np.meshgrid(x, y)

a1 = random.randint(0, 5)
a0 = random.randint(0, 5)

def update(i):
    global a0, a1, a2

    plt.cla()

    a2 = a1
    a1 = a0
    a0 = random.randint(0, 5)

    if a0 == 1 or a0 == 3 or a0 == 5:
        q_table_1[a1, a2] += 1
    if a0 == 0 or a0 == 2 or a0 == 4:
        q_table_2[a1, a2] += 1

    Z1 = q_table_1
    Z2 = q_table_2

    ax = Axes3D(fig)
    
    plt.title('i=' + str(i) + '   [ ' + str(a0+1) + ' ]')

    ax.plot_wireframe(X, Y, Z1, color='b')
    ax.plot_wireframe(X, Y, Z2, color='r')

ani = animation.FuncAnimation(fig, update, interval = 100)
plt.show()
