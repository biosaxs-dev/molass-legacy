"""
    MpInverseBasic.py
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1, 24, 100)

fig, ax = plt.subplots()
ax.set_title(r"$M=P \cdot C$ Metaphor", fontsize=20)
ax.set_aspect("equal")
ax.set_xlabel("C", fontsize=16)
ax.set_ylabel("P", fontsize=16)

ax.plot(x, 24/x, label=r"$P \cdot C=24$")
ax.plot(3, 8, "o", color="red")
ax.plot(5, 8, "o")
ax.plot(5, 4.8, "o")

ax.annotate(r"$(C,P)=(3,8)$", xy=(3,8), xytext=(4,15), arrowprops=dict( arrowstyle="-", color="gray" ), fontsize=16 )
ax.annotate("", xy=(5,8), xytext=(3,8), arrowprops=dict( arrowstyle="->", color="gray" ))
ax.annotate("", xy=(5,4.8), xytext=(5,8), arrowprops=dict( arrowstyle="->", color="gray" ))

ax.legend(fontsize=16)
fig.tight_layout()
plt.show()
