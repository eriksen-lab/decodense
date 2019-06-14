#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

c_core = np.array(list(range(0,11)))
c_single_c = np.array(list(range(11,19))+list(range(29,31)))
c_h = np.array(list(range(19,29))+list(range(31,45)))

res = np.array([
-43.768,
-43.670,
-43.670,
-43.331,
-43.331,
-42.715,
-42.715,
-41.538,
-41.538,
-39.540,
-39.540,
-19.523,
-19.523,
-19.307,
-19.307,
-18.834,
-18.834,
-17.967,
-17.967,
-17.297,
-17.297,
-17.208,
-17.208,
-17.208,
-17.208,
-16.872,
-16.872,
-16.872,
-16.872,
-16.392,
-16.392,
-16.297,
-16.297,
-16.297,
-16.297,
-15.175,
-15.175,
-15.175,
-15.175,
-13.539,
-13.539,
-13.539,
-13.539,
-13.053,
-13.053
])

sns.set(style='darkgrid', font='DejaVu Sans')

fig, ax = plt.subplots()

palette = sns.color_palette('Set2')

plt_c_core = ax.scatter(c_core, res[c_core], \
                        s=150, marker='.', color=palette[3], label='C(1s)')
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
plt.savefig('c11h24_hf.pdf', bbox_inches = 'tight', dpi=1000)

