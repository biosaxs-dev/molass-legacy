"""
    SecTheory.PlateTheory.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
import io
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
import matplotlib.animation as animation
import math

NUM_PLATES = 64
NUM_PARTICLES = 2000
NUM_CYCLES = 200

def update_states(plate_states, p):
    q = 1 - p
    previos_states = plate_states.copy()
    for k in range(NUM_PLATES):
        increase = 0 if k == 0 else p*previos_states[k-1]
        decrease = q*previos_states[k]
        plate_states[k] = increase + decrease

def draw_column(ax, w, h, states_list):
    dh = h/NUM_PLATES

    for k in range(NUM_PLATES):
        y = dh*(NUM_PLATES - 1 - k)
        ax.add_patch(Rectangle(xy=(0, y), width=w,  height=dh, alpha=0.5))

        for plate_states, color, msize in states_list:
            n = int(plate_states[k])
            pos = np.random.uniform(0, 1, (n, 2))
            ax.plot(w*pos[:,0], y + dh*pos[:,1], "o", color=color, markersize=msize)

def demo(save=False):

    plate_states = np.zeros(NUM_PLATES)
    plate_states[0] = NUM_PARTICLES
    p = 0.8*NUM_PLATES/100

    fig = plt.figure(figsize=(9,5))
    fig.suptitle("Plate Theory Simulation N=%d" % NUM_PLATES, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85, wspace=0.5)

    gs = GridSpec(1,3)

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1:3])

    ax2.set_title("Concentration at the End of Column")

    images = []

    el_x = []
    el_y = []
    line, = ax2.plot(el_x, el_y, lw=5)

    ax2.set_xlim(-10, NUM_CYCLES+10)
    ax2.set_ylim(-0.01, 0.11)

    for k in range(NUM_CYCLES):
        update_states(plate_states, p)

        ax1.cla()
        ax1.set_axis_off()
        ax1.set_title("Column State")   # cleared by ax.cla()

        draw_column(ax1, 1, 1, [(plate_states, "red", 1)])

        el_x.append(k)
        el_y.append(plate_states[-1]/NUM_PARTICLES)
        line.set_xdata(el_x)
        line.set_ydata(el_y)

        plt.pause(0.1)

        buffer = io.BytesIO()
        fig.savefig(buffer)
        buffer.seek(0)
        img = mpimg.imread(buffer)
        images.append(img)

    plt.show()
    plt.close()

    if not save:
        return

    fig, ax = plt.subplots(figsize=(9,5))
    ax.set_axis_off()
    fig.tight_layout()

    ims = []
    for img in images:
        im = ax.imshow(img, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
    ani.save("plate_theory.gif")

    plt.show()

def demo2(save=False):

    plate_states1 = np.zeros(NUM_PLATES)
    plate_states2 = np.zeros(NUM_PLATES)
    num_particles1 = NUM_PARTICLES//2
    num_particles2 = NUM_PARTICLES//2
    plate_states1[0] = num_particles1
    plate_states2[0] = num_particles2

    p1 = 0.9*NUM_PLATES/100
    p2 = 0.7*NUM_PLATES/100

    fig = plt.figure(figsize=(9,5))
    fig.suptitle("Plate Theory Simulation N=%d with Two Species" % NUM_PLATES, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85, wspace=0.5)

    gs = GridSpec(1,3)

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1:3])

    ax2.set_title("Concentration at the End of Column")

    images = []

    el_x = []
    el_y = []
    line, = ax2.plot(el_x, el_y, lw=5)

    ax2.set_xlim(-10, NUM_CYCLES+10)
    ax2.set_ylim(-0.01, 0.11)

    for k in range(NUM_CYCLES):
        update_states(plate_states1, p1)
        update_states(plate_states2, p2)

        ax1.cla()
        ax1.set_axis_off()
        ax1.set_title("Column State")   # cleared by ax.cla()

        draw_column(ax1, 1, 1, [(plate_states1, "red", 2), (plate_states2, "green", 1)])

        el_x.append(k)

        y = plate_states1[-1]/num_particles1 + plate_states2[-1]/num_particles2
        el_y.append(y)
        line.set_xdata(el_x)
        line.set_ydata(el_y)

        plt.pause(0.1)

        buffer = io.BytesIO()
        fig.savefig(buffer)
        buffer.seek(0)
        img = mpimg.imread(buffer)
        images.append(img)

    plt.show()
    plt.close()

    if not save:
        return

    print("saving the animation ...")

    fig, ax = plt.subplots(figsize=(9,5))
    ax.set_axis_off()
    fig.tight_layout()

    ims = []
    for img in images:
        print(".")
        im = ax.imshow(img, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
    ani.save("plate_theory.gif")

    plt.show()

def demo3():
    fig, ax = plt.subplots(figsize=(15,4))

    fig.suptitle("Pascal's Triangle in Theoretical Plates", fontsize=16)

    h = 1
    num_plates = 5
    dh = h/num_plates
    w = dh*0.7

    def term(n, k):
        assert n > 0
        m = n - k
        t = r"%d \cdot " % math.comb(n, k)
        if m > 0:
            t += "q"
            if m > 1:
                t += "^%d" % m

        if k > 0:
            t += "p"
            if k > 1:
                t += "^%d" % k
        return t

    ax.set_axis_off()
    for i in range(num_plates):
        x = dh*i + 0.03
        for k in range(num_plates):
            y = dh*(num_plates - 1 - k)
            ax.add_patch(Rectangle(xy=(x, y), width=w,  height=dh, alpha=0.5, edgecolor="blue"))
            if k <= i:
                tx = x + 0.5*w
                ty = y + 0.5*dh
                f = "2000"
                if i > 0:
                    f += r" \cdot "
                    f += term(i, k)
                ax.text(tx, ty, r"$ %s $" % f, ha="center", va="center")
                if i < num_plates - 1:
                    if False:
                        ax.arrow(tx+dh*0.2, ty, dh*0.5, 0, head_width=0.02, head_length=0.01, fc='k', ec='k', alpha=0.5)
                        ax.arrow(tx+dh*0.2, ty, dh*0.5, -dh*0.9, head_width=0.02, head_length=0.01, fc='k', ec='k', alpha=0.5,
                                arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3"),
                                )
                    else:
                        ax.annotate("", xy=(tx+dh*0.8, ty), xycoords='data', xytext=(tx+dh*0.2, ty), textcoords='data',
                                arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3"), alpha=0.5,
                                )
                        ax.annotate("", xy=(tx+dh*0.8, ty -dh*0.94), xycoords='data', xytext=(tx+dh*0.2, ty), textcoords='data',
                                arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3"), alpha=0.5,
                                )

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    # import seaborn
    # seaborn.set()
    # demo2(save=True)
    demo3()
