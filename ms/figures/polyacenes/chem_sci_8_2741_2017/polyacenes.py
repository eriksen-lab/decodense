#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

n_rings = np.arange(2,7)

c_core = np.array([
-44.686,
-47.646,
-50.605,
-52.559,
-54.506
])

c_double_c = np.array([
-38.572,
-43.730,
-49.338,
-53.161,
-57.136
])

c_single_c = np.array([
-19.715,
-23.138,
-26.683,
-27.869,
-30.575
])

c_h = np.array([
-15.490,
-19.091,
-21.338,
-23.589,
-25.209
])

sns.set(style='darkgrid', font='DejaVu Sans')

fig, ax = plt.subplots()

palette = sns.color_palette('Set1')

ax.plot(n_rings, c_core, linewidth=2, linestyle='-', label='C(1s)')
ax.plot(n_rings, c_double_c, linewidth=2, linestyle='-', label='C=C')
ax.plot(n_rings, c_single_c, linewidth=2, linestyle='-', label='C-C')
ax.plot(n_rings, c_h, linewidth=2, linestyle='-', label='C-H')

ax.set_ylim(top=-11.0)
ax.xaxis.grid(False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xticks(n_rings)
ax.set_xlabel('Number of Aromatic Rings')
ax.set_ylabel('Energy Contribution (in au)')
ax.legend(loc='upper right', frameon=False)

sns.despine()
plt.savefig('polyacenes.pdf', bbox_inches = 'tight', dpi=1000)

