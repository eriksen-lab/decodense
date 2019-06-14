#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

c_core = np.array(list(range(0,7)))
c_single_c = np.array(list(range(7,13)))
c_h = np.array(list(range(13,29)))

res = np.array([
-40.882,
-40.660,
-40.660,
-39.761,
-39.761,
-37.975,
-37.975,
-16.591,
-16.591,
-16.056,
-16.056,
-14.725,
-14.725,
-14.448,
-14.448,
-14.261,
-14.261,
-14.261,
-14.261,
-13.407,
-13.407,
-13.407,
-13.407,
-11.986,
-11.986,
-11.986,
-11.986,
-11.570,
-11.569
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
plt.savefig('c7h16_hf.pdf', bbox_inches = 'tight', dpi=1000)

