#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

c_double_c = np.array(list(range(0,5))+list(range(11,15))+list(range(35,39)))
c_core = np.array(list(range(5,11))+list(range(15,35)))
c_single_c = np.array(list(range(35,55))+list(range(63,65)))
c_h = np.array(list(range(55,63))+list(range(65,73)))

res = np.array([
-57.136,
-56.482,
-56.482,
-56.346,
-56.346,
-54.506,
-54.506,
-54.008,
-54.008,
-54.008,
-54.008,
-53.784,
-53.784,
-53.644,
-53.644,
-52.836,
-52.836,
-52.836,
-52.836,
-52.209,
-52.209,
-52.209,
-52.209,
-51.799,
-51.799,
-51.799,
-51.799,
-48.978,
-48.978,
-48.978,
-48.978,
-47.426,
-47.426,
-47.426,
-47.426,
-46.680,
-46.680,
-46.518,
-46.518,
-30.575,
-30.079,
-30.079,
-29.644,
-29.644,
-29.644,
-28.862,
-28.862,
-28.862,
-28.862,
-28.275,
-28.275,
-26.535,
-26.535,
-26.535,
-26.535,
-25.209,
-25.209,
-25.209,
-25.209,
-24.227,
-24.227,
-24.227,
-24.227,
-23.543,
-23.543,
-21.650,
-21.650,
-21.650,
-21.650,
-19.722,
-19.722,
-19.722,
-19.722
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
plt.savefig('c26h16_hf.pdf', bbox_inches = 'tight', dpi=1000)

