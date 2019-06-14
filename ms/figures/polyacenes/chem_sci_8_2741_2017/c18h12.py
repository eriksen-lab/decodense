#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

c_double_c = np.array([6]+list(range(11,15))+list(range(23,27)))
c_core = np.array(list(range(0,6))+list(range(7,11))+list(range(15,23)))
c_single_c = np.array(list(range(27,39)))
c_h = np.array(list(range(39,51)))

res = np.array([
-50.605,
-50.605,
-49.601,
-49.601,
-49.601,
-49.601,
-49.338,
-48.875,
-48.875,
-48.875,
-48.875,
-48.179,
-48.179,
-48.136,
-48.136,
-46.660,
-46.660,
-46.660,
-46.660,
-45.307,
-45.307,
-45.307,
-45.307,
-42.268,
-42.268,
-42.061,
-42.061,
-26.683,
-25.721,
-25.721,
-25.721,
-25.669,
-25.669,
-24.078,
-24.078,
-24.078,
-24.078,
-21.426,
-21.426,
-21.338,
-21.338,
-21.338,
-21.338,
-19.349,
-19.349,
-19.349,
-19.349,
-17.702,
-17.702,
-17.702,
-17.702
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
plt.savefig('c18h12_hf.pdf', bbox_inches = 'tight', dpi=1000)

