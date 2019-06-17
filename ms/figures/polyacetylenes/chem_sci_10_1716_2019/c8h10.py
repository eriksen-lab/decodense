#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

c_double_c = np.array(list(range(8,12)))
c_core = np.array(list(range(0,8)))
c_single_c = np.array(list(range(12,15)))
c_h = np.array(list(range(15,25)))

res = np.array([
-40.552,
-40.552,
-40.203,
-40.203,
-39.289,
-39.289,
-37.545,
-37.545,
-30.779,
-30.779,
-27.149,
-27.149,
-16.495,
-15.704,
-15.704,
-14.096,
-14.096,
-13.775,
-13.775,
-12.916,
-12.916,
-11.555,
-11.555,
-11.140,
-11.140
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
plt.savefig('c8h10_hf.pdf', bbox_inches = 'tight', dpi=1000)

