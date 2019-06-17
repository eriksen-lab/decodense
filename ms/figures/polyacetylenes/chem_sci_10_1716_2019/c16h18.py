#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

c_double_c = np.array(list(range(16,24)))
c_core = np.array(list(range(0,16)))
c_single_c = np.array(list(range(24,31)))
c_h = np.array(list(range(31,49)))

res = np.array([
-44.613,
-44.613,
-44.522,
-44.522,
-44.328,
-44.328,
-44.020,
-44.020,
-43.539,
-43.539,
-42.842,
-42.842,
-41.648,
-41.648,
-39.688,
-39.688,
-39.093,
-39.093,
-38.318,
-38.318,
-36.395,
-36.395,
-31.656,
-31.656,
-20.561,
-20.373,
-20.373,
-19.725,
-19.725,
-18.203,
-18.203,
-18.110,
-18.110,
-18.020,
-18.020,
-17.831,
-17.831,
-17.528,
-17.528,
-17.063,
-17.063,
-16.398,
-16.398,
-15.270,
-15.270,
-13.684,
-13.684,
-13.189,
-13.189
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
plt.savefig('c16h18_hf.pdf', bbox_inches = 'tight', dpi=1000)

