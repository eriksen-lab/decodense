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

c18h12_hf_pm = np.array([
-50.598, 
-50.598,
-49.593,
-49.593,
-49.593,
-49.593,
-48.869,
-48.869,
-48.869,
-48.869,
-46.653,
-46.653,
-46.653,
-46.653,
-45.301,
-45.301,
-45.301,
-45.301,
-26.697,
-25.727,
-25.727,
-25.727,
-25.727,
-25.683,
-25.683,
-25.292,
-25.292,
-25.292,
-25.292,
-24.084,
-24.084,
-24.084,
-24.084,
-23.768,
-22.873,
-22.873,
-22.858,
-22.858,
-22.230,
-22.230,
-22.230,
-22.230,
-21.438,
-21.438,
-21.331,
-21.331,
-21.331,
-21.331,
-19.932,
-19.932,
-19.911,
-19.911,
-19.342,
-19.342,
-19.342,
-19.342,
-17.694,
-17.694,
-17.694,
-17.694
])

n_core_c18h12 = 18 # +4
n_c_c_single_c18h12 = 21 # +5
n_c_c_double_c18h12 = 9 # +2
n_c_h_c18h12 = 12 # +2

sns.set(style='darkgrid', font='DejaVu Sans')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

palette = sns.color_palette('Set2')
palette = [palette[1]] * len(palette)

c18h12_pm_core_fake = ax1.scatter(n_core_c18h12, \
                             c18h12_hf_pm[n_core_c18h12], \
                             s=150, marker='.', color=palette[3])
c18h12_pm_c_c_single = ax1.scatter(np.arange(n_core_c18h12, (n_core_c18h12 + n_c_c_single_c18h12)), \
                              c18h12_hf_pm[n_core_c18h12:(n_core_c18h12 + n_c_c_single_c18h12)], \
                              s=150, marker='.', color=palette[0])
c18h12_pm_c_c_double = ax1.scatter(np.arange((n_core_c18h12 + n_c_c_single_c18h12), (n_core_c18h12 + n_c_c_single_c18h12 + n_c_c_double_c18h12)), \
                              c18h12_hf_pm[(n_core_c18h12 + n_c_c_single_c18h12):(n_core_c18h12 + n_c_c_single_c18h12 + n_c_c_double_c18h12)], \
                              s=150, marker='.', color=palette[1])
c18h12_pm_c_h = ax1.scatter(np.arange((n_core_c18h12 + n_c_c_single_c18h12 + n_c_c_double_c18h12), (n_core_c18h12 + n_c_c_single_c18h12 + n_c_c_double_c18h12 + n_c_h_c18h12)), \
                       c18h12_hf_pm[(n_core_c18h12 + n_c_c_single_c18h12 + n_c_c_double_c18h12):(n_core_c18h12 + n_c_c_single_c18h12 + n_c_c_double_c18h12 + n_c_h_c18h12)], \
                       s=150, marker='.', color=palette[2])
ax1.xaxis.grid(False)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

ax2.scatter(np.arange(n_core_c18h12), \
            c18h12_hf_pm[:n_core_c18h12], \
            s=150, marker='.', color=palette[3])
ax2.xaxis.grid(False)
ax2.legend([(c18h12_pm_core_fake, c18h12_pm_c_c_single, c18h12_pm_c_c_double, c18h12_pm_c_h)], \
           ['Tetracene (C$_{18}$H$_{12}$)'], loc='lower right', \
           numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax2.set_xlabel('MO Index')

fig.text(0.0, 0.5, 'MO Contribution (in au)', ha='center', va='center', rotation='vertical')

fig.subplots_adjust(hspace=0.05)
sns.despine()
plt.savefig('c18h12_hf.pdf', bbox_inches = 'tight', dpi=1000)

