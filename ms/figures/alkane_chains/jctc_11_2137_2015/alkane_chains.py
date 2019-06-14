#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

n_carbons = np.arange(6,12)

c_core = np.array([
-39.857,
-40.882,
-41.685,
-42.488,
-43.128,
-43.768
])

c_single_c = np.array([
-15.683,
-16.591,
-17.500,
-18.216,
-18.933,
-19.523
])

c_h = np.array([
-13.465,
-14.448,
-15.244,
-16.040,
-16.669,
-17.297
])

sns.set(style='darkgrid', font='DejaVu Sans')

fig, ax = plt.subplots()

palette = sns.color_palette('Set1')

ax.plot(n_carbons, c_core, linewidth=2, linestyle='-', label='C(1s)')
ax.plot(n_carbons, c_single_c, linewidth=2, linestyle='-', label='C-C')
ax.plot(n_carbons, c_h, linewidth=2, linestyle='-', label='C-H')

ax.set_ylim(top=-8.0)
ax.xaxis.grid(False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xticks(n_carbons)
ax.set_xlabel('Number of Carbons')
ax.set_ylabel('Energy Contribution (in au)')
ax.legend(loc='upper right', frameon=False)

sns.despine()
plt.savefig('alkane_chains.pdf', bbox_inches = 'tight', dpi=1000)

