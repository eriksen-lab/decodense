#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import seaborn as sns

c6h6_hf_can = np.array([
-38.914,
-38.914,
-38.914,
-38.913,
-38.913,
-38.913,
-15.456,
-14.656,
-14.656,
-14.334,
-13.628,
-13.628,
-13.066,
-12.854,
-12.854,
-12.822,
-12.739,
-12.739,
-12.675,
-12.675,
-11.779
])

c6h6_hf_pm = np.array([
-38.913,
-38.913,
-38.913,
-38.913,
-38.913,
-38.913,
-15.121,
-15.121,
-15.121,
-15.121,
-15.121,
-15.121,
-12.805,
-12.805,
-12.805,
-11.903,
-11.903,
-11.903,
-11.903,
-11.903,
-11.903
])

c6h6_hf_ibo = np.array([
-38.913,
-38.913,
-38.913,
-38.913,
-38.913,
-38.913,
-15.104,
-15.104,
-15.104,
-15.104,
-15.104,
-15.104,
-12.805,
-12.805,
-12.805,
-11.920,
-11.920,
-11.920,
-11.920,
-11.920,
-11.920
])

n_core = 6

sns.set(style='darkgrid', font='DejaVu Sans')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

palette = sns.color_palette('Set2')

ax1.scatter(np.arange(n_core, c6h6_hf_can.size), \
                    c6h6_hf_can[n_core:], \
                    s=150, marker='.', color='black', label='Canonical')
#ax1.scatter(np.arange(n_core, c6h6_hf_pm.size), \
#                    c6h6_hf_pm[n_core:], \
#                    s=150, marker='.', color=palette[0], label='Pipek-Mezey')
ax1.scatter(np.arange(n_core, c6h6_hf_ibo.size), \
                    c6h6_hf_ibo[n_core:], \
                    s=150, marker='.', color=palette[1], label='IBOs')
ax1.xaxis.grid(False)
ax1.legend(loc='upper left', frameon=False)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

ax2.scatter(np.arange(n_core), \
            c6h6_hf_can[:n_core], \
            s=150, marker='.', color='black')
#ax2.scatter(np.arange(n_core), \
#            c6h6_hf_pm[:n_core], \
#            s=150, marker='.', color=palette[0])
ax2.scatter(np.arange(n_core), \
            c6h6_hf_ibo[:n_core], \
            s=150, marker='.', color=palette[1])
ax2.xaxis.grid(False)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax2.set_xlabel('MO Index')

fig.text(0.0, 0.5, 'MO Contribution (in au)', ha='center', va='center', rotation='vertical')

fig.subplots_adjust(hspace=0.05)
sns.despine()
plt.savefig('c6h6_hf.pdf', bbox_inches = 'tight', dpi=1000)

