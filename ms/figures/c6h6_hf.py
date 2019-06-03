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
-38.907,
-38.907,
-38.907,
-38.907,
-38.907,
-38.907,
-15.126,
-15.126,
-15.126,
-15.126,
-15.126,
-15.126,
-12.805,
-12.805,
-12.805,
-11.905,
-11.905,
-11.905,
-11.905,
-11.905,
-11.905
])

n_core = 6
n_c_c_single = 6
n_c_c_double = 3
n_c_h = 6

sns.set(style='darkgrid', font='DejaVu Sans')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

palette = sns.color_palette('Set2')

c6h6_can = ax1.scatter(np.arange(n_core, c6h6_hf_can.size), \
                    c6h6_hf_can[n_core:], \
                    s=150, marker='.', color='black')
c6h6_pm_core_fake = ax1.scatter(n_core, \
                             c6h6_hf_pm[n_core], \
                             s=150, marker='.', color=palette[3])
c6h6_pm_c_c_single = ax1.scatter(np.arange(n_core, (n_core + n_c_c_single)), \
                              c6h6_hf_pm[n_core:(n_core + n_c_c_single)], \
                              s=150, marker='.', color=palette[0])
c6h6_pm_c_c_double = ax1.scatter(np.arange((n_core + n_c_c_single), (n_core + n_c_c_single + n_c_c_double)), \
                              c6h6_hf_pm[(n_core + n_c_c_single):(n_core + n_c_c_single + n_c_c_double)], \
                              s=150, marker='.', color=palette[1])
c6h6_pm_c_h = ax1.scatter(np.arange((n_core + n_c_c_single + n_c_c_double), (n_core + n_c_c_single + n_c_c_double + n_c_h)), \
                       c6h6_hf_pm[(n_core + n_c_c_single + n_c_c_double):(n_core + n_c_c_single + n_c_c_double + n_c_h)], \
                       s=150, marker='.', color=palette[2])
#ax1.text(4.0, -15.25, 'C$-$C', color=palette[0])
#ax1.text(9.6, -12.93, 'C$=$C', color=palette[1])
#ax1.text(12.8, -12.02, 'C$-$H', color=palette[2])
ax1.xaxis.grid(False)
ax1.legend([c6h6_can, (c6h6_pm_core_fake, c6h6_pm_c_c_single, c6h6_pm_c_c_double, c6h6_pm_c_h)], \
           ['Canonical', 'Pipek-Mezey'], loc='upper left', \
           numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

ax2.scatter(np.arange(n_core), \
            c6h6_hf_can[:n_core], \
            s=150, marker='.', color='black')
ax2.scatter(np.arange(n_core), \
            c6h6_hf_pm[:n_core], \
            s=150, marker='.', color=palette[3])
#ax2.text(5.3, -38.91388, 'C(1s)', color='black')
#ax2.text(5.3, -38.90725, 'C(1s)', color=palette[3])
ax2.xaxis.grid(False)
ax2.set_ylim([-38.9145, -38.9055])
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax2.set_xlabel('MO Index')

fig.text(0.0, 0.5, 'MO Contribution (in au)', ha='center', va='center', rotation='vertical')

fig.subplots_adjust(hspace=0.05)
sns.despine()
plt.savefig('c6h6_hf.pdf', bbox_inches = 'tight', dpi=1000)

