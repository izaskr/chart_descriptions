import matplotlib.pyplot as plt
import numpy as np

x = ["W", "Z", "Q"]
x_pos = [i for i in range(len(x))]
y1 = [5, 16, 8]
y2 = [18, 12, 4]

t1 = "m: present data"
t2 = "m: inverse"
t3 = "m: proportional"

colors = ["lightpink", "steelblue", "tomato"]

fig, axs = plt.subplots(2, 2)
axs[0,0].bar(x_pos, y1, tick_label=x, color=colors)
axs[0, 0].set_title(t1)

axs[0,1].bar(x_pos, y1, tick_label=x, color=colors)
axs[0, 1].set_title(t2)

axs[1,0].bar(x_pos, y2, tick_label=x, color=colors)
axs[1, 0].set_title(t1)

axs[1,1].bar(x_pos, y1, tick_label=x, color=colors)
axs[1, 1].set_title(t3)

for ax in axs.flat:
    ax.label_outer()

fig.savefig("/home/iza/dummy.png")
