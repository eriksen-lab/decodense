#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

c_core = np.array(list(range(0,9)))
c_single_c = np.array(list(range(9,15))+list(range(21,23)))
c_h = np.array(list(range(15,21))+list(range(23,37)))

res = np.array([
-42.488,
-42.325,
-42.325,
-41.842,
-41.842,
-40.766,
-40.766,
-38.849,
-38.849,
-18.216,
-18.216,
-17.897,
-17.897,
-17.146,
-17.146,
-16.040,
-16.040,
-15.873,
-15.873,
-15.872,
-15.872,
-15.661,
-15.661,
-15.430,
-15.430,
-15.430,
-15.430,
-14.406,
-14.406,
-14.406,
-14.406,
-12.851,
-12.851,
-12.851,
-12.851,
-12.393,
-12.393
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
plt.savefig('c9h20_hf.pdf', bbox_inches = 'tight', dpi=1000)

