#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

n_carbons = np.arange(2,17,2)

c_core = np.array([
-33.021,
-36.453,
-38.901,
-40.552,
-41.855,
-42.920,
-43.827,
-44.613
])

c_double_c = np.array([
-16.649,
-22.193,
-27.861,
-30.779,
-33.693,
-35.655,
-37.616,
-39.093
])

c_single_c = np.array([
-12.218,
-14.523,
-16.495,
-17.679,
-18.865,
-19.712,
-20.561
])

c_h = np.array([
-7.135,
-10.172,
-12.495,
-14.096,
-15.377,
-16.430,
-17.328,
-18.110
])

sns.set(style='darkgrid', font='DejaVu Sans')

fig, ax = plt.subplots()

palette = sns.color_palette('Set1')

ax.plot(n_carbons, c_core, linewidth=2, linestyle='-', label='C(1s)')
ax.plot(n_carbons, c_double_c, linewidth=2, linestyle='-', label='C=C')
ax.plot(n_carbons[1:], c_single_c, linewidth=2, linestyle='-', label='C-C')
ax.plot(n_carbons, c_h, linewidth=2, linestyle='-', label='C-H')

ax.set_ylim(top=-3.0)
ax.xaxis.grid(False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xticks(n_carbons)
ax.set_xlabel('Number of Carbon Atoms')
ax.set_ylabel('Energy Contribution (in au)')
ax.legend(loc='upper right', frameon=False)

sns.despine()
plt.savefig('polyacetylenes.pdf', bbox_inches = 'tight', dpi=1000)

