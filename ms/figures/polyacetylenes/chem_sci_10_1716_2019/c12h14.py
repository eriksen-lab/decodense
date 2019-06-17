#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

c_double_c = np.array(list(range(12,18)))
c_core = np.array(list(range(0,12)))
c_single_c = np.array(list(range(18,23)))
c_h = np.array(list(range(23,37)))

res = np.array([
-42.920,
-42.920,
-42.761,
-42.761,
-42.402,
-42.402,
-41.803,
-41.803,
-40.695,
-40.695,
-38.804,
-38.804,
-35.655,
-35.655,
-34.220,
-34.220,
-29.816,
-29.816,
-18.865,
-18.526,
-18.526,
-17.207,
-17.207,
-16.430,
-16.430,
-16.275,
-16.275,
-15.929,
-15.929,
-15.363,
-15.363,
-14.317,
-14.317,
-12.804,
-12.804,
-12.336,
-12.336
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
plt.savefig('c12h14_hf.pdf', bbox_inches = 'tight', dpi=1000)

