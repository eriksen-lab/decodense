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

c10h8_hf_can = np.array([
-44.673,
-44.669,
-42.571,
-42.568,
-42.318,
-42.312,
-42.303,
-42.284,
-42.047,
-42.033,
-19.759,
-18.835,
-18.635,
-18.583,
-18.085,
-17.823,
-17.708,
-17.638,
-17.549,
-17.460,
-17.307,
-17.016,
-16.836,
-16.820,
-16.731,
-16.652,
-16.574,
-16.542,
-16.294,
-16.116,
-16.068,
-16.050,
-15.979,
-15.673
])

c10h8_hf_pm = np.array([
-44.678,
-44.678,
-42.709,
-42.709,
-42.709,
-42.709,
-41.879,
-41.879,
-41.879,
-41.879,
-20.796,
-19.722,
-19.722,
-19.722,
-19.722,
-18.531,
-18.531,
-18.531,
-18.531,
-18.047,
-18.047,
-17.913,
-16.236,
-16.236,
-16.236,
-16.236,
-15.484,
-15.484,
-15.484,
-15.484,
-14.527,
-14.527,
-14.527,
-14.527
])

n_core_c6h6 = 6
n_c_c_single_c6h6 = 6
n_c_c_double_c6h6 = 3
n_c_h_c6h6 = 6
n_core_c10h8 = 10
n_c_c_single_c10h8 = 11
n_c_c_double_c10h8 = 5
n_c_h_c10h8 = 8

sns.set(style='darkgrid', font='DejaVu Sans')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

palette = sns.color_palette('Set2')

c6h6_pm_core_fake = ax1.scatter(n_core_c6h6, \
                             c6h6_hf_pm[n_core_c6h6], \
                             s=30, marker='x', color=palette[3])
c6h6_pm_c_c_single = ax1.scatter(np.arange(n_core_c6h6, (n_core_c6h6 + n_c_c_single_c6h6)), \
                              c6h6_hf_pm[n_core_c6h6:(n_core_c6h6 + n_c_c_single_c6h6)], \
                              s=30, marker='x', color=palette[0])
c6h6_pm_c_c_double = ax1.scatter(np.arange((n_core_c6h6 + n_c_c_single_c6h6), (n_core_c6h6 + n_c_c_single_c6h6 + n_c_c_double_c6h6)), \
                              c6h6_hf_pm[(n_core_c6h6 + n_c_c_single_c6h6):(n_core_c6h6 + n_c_c_single_c6h6 + n_c_c_double_c6h6)], \
                              s=30, marker='x', color=palette[1])
c6h6_pm_c_h = ax1.scatter(np.arange((n_core_c6h6 + n_c_c_single_c6h6 + n_c_c_double_c6h6), (n_core_c6h6 + n_c_c_single_c6h6 + n_c_c_double_c6h6 + n_c_h_c6h6)), \
                       c6h6_hf_pm[(n_core_c6h6 + n_c_c_single_c6h6 + n_c_c_double_c6h6):(n_core_c6h6 + n_c_c_single_c6h6 + n_c_c_double_c6h6 + n_c_h_c6h6)], \
                       s=30, marker='x', color=palette[2])
c10h8_pm_core_fake = ax1.scatter(n_core_c10h8, \
                             c10h8_hf_pm[n_core_c10h8], \
                             s=150, marker='.', color=palette[3])
c10h8_pm_c_c_single = ax1.scatter(np.arange(n_core_c10h8, (n_core_c10h8 + n_c_c_single_c10h8)), \
                              c10h8_hf_pm[n_core_c10h8:(n_core_c10h8 + n_c_c_single_c10h8)], \
                              s=150, marker='.', color=palette[0])
c10h8_pm_c_c_double = ax1.scatter(np.arange((n_core_c10h8 + n_c_c_single_c10h8), (n_core_c10h8 + n_c_c_single_c10h8 + n_c_c_double_c10h8)), \
                              c10h8_hf_pm[(n_core_c10h8 + n_c_c_single_c10h8):(n_core_c10h8 + n_c_c_single_c10h8 + n_c_c_double_c10h8)], \
                              s=150, marker='.', color=palette[1])
c10h8_pm_c_h = ax1.scatter(np.arange((n_core_c10h8 + n_c_c_single_c10h8 + n_c_c_double_c10h8), (n_core_c10h8 + n_c_c_single_c10h8 + n_c_c_double_c10h8 + n_c_h_c10h8)), \
                       c10h8_hf_pm[(n_core_c10h8 + n_c_c_single_c10h8 + n_c_c_double_c10h8):(n_core_c10h8 + n_c_c_single_c10h8 + n_c_c_double_c10h8 + n_c_h_c10h8)], \
                       s=150, marker='.', color=palette[2])
ax1.xaxis.grid(False)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

ax2.scatter(np.arange(n_core_c6h6), \
            c6h6_hf_can[:n_core_c6h6], \
            s=30, marker='x', color=palette[3])
ax2.scatter(np.arange(n_core_c10h8), \
            c10h8_hf_pm[:n_core_c10h8], \
            s=150, marker='.', color=palette[3])
ax2.xaxis.grid(False)
ax2.legend([(c6h6_pm_core_fake, c6h6_pm_c_c_single, c6h6_pm_c_c_double, c6h6_pm_c_h), (c10h8_pm_core_fake, c10h8_pm_c_c_single, c10h8_pm_c_c_double, c10h8_pm_c_h)], \
           ['Benzene', 'Naphthalene'], loc='lower right', \
           numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax2.set_xlabel('MO Index')

fig.text(0.0, 0.5, 'MO Contribution (in au)', ha='center', va='center', rotation='vertical')

fig.subplots_adjust(hspace=0.05)
sns.despine()
plt.savefig('c10h8_hf.pdf', bbox_inches = 'tight', dpi=1000)

