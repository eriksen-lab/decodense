#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

c_core = np.array(list(range(0,10)))
c_single_c = np.array(list(range(10,17))+list(range(25,27)))
c_h = np.array(list(range(17,25))+list(range(27,41)))

res = np.array([
-43.128,
-43.128,
-42.867,
-42.867,
-42.306,
-42.306,
-41.175,
-41.175,
-39.212,
-39.212,
-18.933,
-18.806,
-18.806,
-18.399,
-18.399,
-17.582,
-17.582,
-16.669,
-16.669,
-16.669,
-16.669,
-16.412,
-16.412,
-16.412,
-16.412,
-16.047,
-16.047,
-15.889,
-15.889,
-15.889,
-15.889,
-14.814,
-14.814,
-14.814,
-14.814,
-13.212,
-13.212,
-13.212,
-13.212,
-12.740,
-12.740
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
plt.savefig('c10h22_hf.pdf', bbox_inches = 'tight', dpi=1000)

