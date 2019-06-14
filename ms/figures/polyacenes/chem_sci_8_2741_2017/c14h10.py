#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

c_double_c = np.array(list(range(10,12))+list(range(16,21)))
c_core = np.array(list(range(0,10))+list(range(12,16)))
c_single_c = np.array(list(range(21,30)))
c_h = np.array(list(range(30,40)))

res = np.array([
-47.646,
-47.646,
-47.646,
-47.646,
-46.562,
-46.562,
-45.017,
-45.017,
-45.017,
-45.017,
-43.932,
-43.932,
-43.848,
-43.848,
-43.848,
-43.848,
-43.730,
-39.193,
-39.193,
-38.931,
-38.931,
-23.727,
-23.727,
-23.138,
-22.297,
-22.297,
-22.297,
-22.297,
-19.978,
-19.978,
-19.091,
-19.091,
-17.729,
-17.729,
-17.729,
-17.729,
-16.335,
-16.335,
-16.335,
-16.335
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
plt.savefig('c14h10_hf.pdf', bbox_inches = 'tight', dpi=1000)

