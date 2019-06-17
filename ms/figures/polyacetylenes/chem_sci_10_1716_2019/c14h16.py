#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

c_double_c = np.array(list(range(14,21)))
c_core = np.array(list(range(0,14)))
c_single_c = np.array(list(range(21,27)))
c_h = np.array(list(range(27,43)))

res = np.array([
-43.827,
-43.827,
-43.706,
-43.706,
-43.456,
-43.456,
-43.024,
-43.024,
-42.367,
-42.367,
-41.209,
-41.209,
-39.279,
-39.279,
-37.616,
-37.133,
-37.133,
-35.405,
-35.405,
-30.806,
-30.806,
-19.712,
-19.712,
-19.185,
-19.185,
-17.746,
-17.746,
-17.328,
-17.328,
-17.211,
-17.211,
-16.966,
-16.966,
-16.549,
-16.549,
-15.925,
-15.925,
-14.831,
-14.831,
-13.276,
-13.276,
-12.793,
-12.793
])

sns.set(style='darkgrid', font='DejaVu Sans')

fig, ax = plt.subplots()

palette = sns.color_palette('Set2')

plt_c_core = ax.scatter(c_core, res[c_core], \
                        s=150, marker='.', color=palette[3], label='C(1s)')
plt_c_double_c = ax.scatter(c_double_c, res[c_double_c], \
                            s=150, marker='.', color=palette[0], label='C=C')
plt_c_single_c = ax.scatter(c_single_c, res[c_single_c], \
                            s=150, marker='.', color=palette[1], label='C-C')
plt_c_h = ax.scatter(c_h, res[c_h], \
                     s=150, marker='.', color=palette[2], label='C-H')

ax.xaxis.grid(False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.set_xlabel('Contribution')
ax.set_ylabel('Energy Contribution (in au)')
ax.legend(loc='lower right', frameon=False)

sns.despine()
plt.savefig('c14h16_hf.pdf', bbox_inches = 'tight', dpi=1000)

