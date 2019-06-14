#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

c_core = np.array(list(range(0,8)))
c_single_c = np.array(list(range(8,13))+list(range(17,19)))
c_h = np.array(list(range(13,17))+list(range(19,33)))

res = np.array([
-41.685,
-41.685,
-41.300,
-41.300,
-40.303,
-40.303,
-38.439,
-38.439,
-17.500,
-17.308,
-17.308,
-16.645,
-16.645,
-15.244,
-15.244,
-15.244,
-15.244,
-15.226,
-15.226,
-14.890,
-14.890,
-14.890,
-14.890,
-13.947,
-13.947,
-13.947,
-13.947,
-12.444,
-12.444,
-12.444,
-12.444,
-12.007,
-12.007
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
plt.savefig('c8h18_hf.pdf', bbox_inches = 'tight', dpi=1000)

