#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

c14h10_hf_pm = np.array([
-47.639, 
-47.639,
-47.639,
-47.639,
-46.555,
-46.555,
-45.010,
-45.010,
-45.010,
-45.010,
-43.842,
-43.842,
-43.842,
-43.842,
-23.740,
-23.740,
-23.145,
-23.145,
-23.145,
-23.145,
-22.303,
-22.303,
-22.303,
-22.303,
-20.826,
-20.826,
-20.674,
-20.674,
-20.674,
-20.674,
-20.644,
-19.989,
-19.989,
-19.085,
-19.085,
-18.431,
-18.431,
-18.318,
-18.318,
-17.723,
-17.723,
-17.723,
-17.723,
-16.327,
-16.327,
-16.327,
-16.327
])

n_core_c14h10 = 14
n_c_c_single_c14h10 = 16
n_c_c_double_c14h10 = 7
n_c_h_c14h10 = 10

sns.set(style='darkgrid', font='DejaVu Sans')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

palette = sns.color_palette('Set2')
palette = [palette[1]] * len(palette)

c14h10_pm_core_fake = ax1.scatter(n_core_c14h10, \
                             c14h10_hf_pm[n_core_c14h10], \
                             s=150, marker='.', color=palette[3])
c14h10_pm_c_c_single = ax1.scatter(np.arange(n_core_c14h10, (n_core_c14h10 + n_c_c_single_c14h10)), \
                              c14h10_hf_pm[n_core_c14h10:(n_core_c14h10 + n_c_c_single_c14h10)], \
                              s=150, marker='.', color=palette[0])
c14h10_pm_c_c_double = ax1.scatter(np.arange((n_core_c14h10 + n_c_c_single_c14h10), (n_core_c14h10 + n_c_c_single_c14h10 + n_c_c_double_c14h10)), \
                              c14h10_hf_pm[(n_core_c14h10 + n_c_c_single_c14h10):(n_core_c14h10 + n_c_c_single_c14h10 + n_c_c_double_c14h10)], \
                              s=150, marker='.', color=palette[1])
c14h10_pm_c_h = ax1.scatter(np.arange((n_core_c14h10 + n_c_c_single_c14h10 + n_c_c_double_c14h10), (n_core_c14h10 + n_c_c_single_c14h10 + n_c_c_double_c14h10 + n_c_h_c14h10)), \
                       c14h10_hf_pm[(n_core_c14h10 + n_c_c_single_c14h10 + n_c_c_double_c14h10):(n_core_c14h10 + n_c_c_single_c14h10 + n_c_c_double_c14h10 + n_c_h_c14h10)], \
                       s=150, marker='.', color=palette[2])
ax1.xaxis.grid(False)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

ax2.scatter(np.arange(n_core_c14h10), \
            c14h10_hf_pm[:n_core_c14h10], \
            s=150, marker='.', color=palette[3])
ax2.xaxis.grid(False)
ax2.legend([(c14h10_pm_core_fake, c14h10_pm_c_c_single, c14h10_pm_c_c_double, c14h10_pm_c_h)], \
           ['Anthracene (C$_{14}$H$_{10}$)'], loc='lower right', \
           numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax2.set_xlabel('MO Index')

fig.text(0.0, 0.5, 'MO Contribution (in au)', ha='center', va='center', rotation='vertical')

fig.subplots_adjust(hspace=0.05)
sns.despine()
plt.savefig('c14h10_hf.pdf', bbox_inches = 'tight', dpi=1000)

