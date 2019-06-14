#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

c_core = np.array(list(range(0,6)))
c_single_c = np.array(list(range(6,11)))
c_h = np.array(list(range(11,25)))

res = np.array([
-39.857,
-39.857,
-39.121,
-39.121,
-37.433,
-37.433,
-15.683,
-15.339,
-15.339,
-14.136,
-14.136,
-13.465,
-13.465,
-13.465,
-13.465,
-12.778,
-12.778,
-12.778,
-12.778,
-11.447,
-11.447,
-11.447,
-11.447,
-11.066,
-11.066
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
plt.savefig('c6h14_hf.pdf', bbox_inches = 'tight', dpi=1000)

