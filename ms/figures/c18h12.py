#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

hf_ibo = np.sort(np.array([
# chem_sci_8_2741_2017: 631g (ibo)
-48.875,
-48.179,
-25.721,
-21.338,
-49.601,
-25.669,
-24.078,
-49.601,
-48.136,
-24.078,
-48.875,
-25.721,
-21.338,
-50.605,
-26.683,
-49.338,
-50.605,
-25.721,
-48.875,
-48.179,
-21.338,
-49.601,
-25.669,
-24.078,
-49.601,
-48.136,
-24.078,
-48.875,
-21.338,
-46.660,
-42.268,
-19.349,
-45.307,
-21.426,
-17.702,
-45.307,
-42.061,
-17.702,
-46.660,
-19.349,
-46.660,
-42.268,
-19.349,
-45.307,
-21.426,
-17.702,
-45.307,
-42.061,
-17.702,
-46.660,
-19.349
# chem_sci_8_2741_2017: 631g (ibo)
#-48.875,
#-48.146,
#-25.722,
#-21.330,
#-49.601,
#-25.679,
#-24.080,
#-49.601,
#-48.158,
#-24.079,
#-48.875,
#-25.722,
#-21.330,
#-50.605,
#-50.463,
#-25.722,
#-50.605,
#-25.722,
#-48.875,
#-48.146,
#-21.330,
#-49.601,
#-25.679,
#-24.080,
#-49.601,
#-48.158,
#-24.080,
#-48.875,
#-21.330,
#-46.660,
#-42.138,
#-19.340,
#-45.307,
#-21.433,
#-17.692,
#-45.307,
#-42.154,
#-17.692,
#-46.660,
#-19.340,
#-46.660,
#-42.137,
#-19.340,
#-45.307,
#-21.433,
#-17.692,
#-45.307,
#-42.154,
#-17.692,
#-46.660,
#-19.340
]))

n_rings = 4

n_core = 6
n_c_c_double = 3
n_c_c_single = 3
n_c_h = 6
if n_rings > 1:
    n_core += (n_rings-1) * 4
    n_c_c_double += (n_rings-1) * 2
    n_c_c_single += (n_rings-1) * 3
    n_c_h += (n_rings-1) * 2

sns.set(style='darkgrid', font='DejaVu Sans')

fig, ax = plt.subplots()

palette = sns.color_palette('Set2')

hf_ibo_core = ax.scatter(np.arange(n_core), \
                         hf_ibo[:n_core], \
                         s=150, marker='.', color=palette[3], label='C(1s)')
hf_ibo_c_c_double = ax.scatter(np.arange(n_core, \
                                         (n_core + n_c_c_double)), \
                               hf_ibo[n_core:(n_core + n_c_c_double)], \
                               s=150, marker='.', color=palette[0], label='C=C')
hf_ibo_c_c_single = ax.scatter(np.arange((n_core + n_c_c_double), \
                                         (n_core + n_c_c_double + n_c_c_single)), \
                               hf_ibo[(n_core + n_c_c_double):(n_core + n_c_c_double + n_c_c_single)], \
                               s=150, marker='.', color=palette[1], label='C-C')
hf_ibo_c_h = ax.scatter(np.arange((n_core + n_c_c_double + n_c_c_single), \
                                  (n_core + n_c_c_double + n_c_c_single + n_c_h)), \
                        hf_ibo[(n_core + n_c_c_double + n_c_c_single):(n_core + n_c_c_double + n_c_c_single + n_c_h)], \
                        s=150, marker='.', color=palette[2], label='C-H')

ax.xaxis.grid(False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.set_xlabel('Contribution')
ax.set_ylabel('Energy Contribution (in au)')
#ax.set_ylim([-49.0, -44.0])
ax.legend(loc='lower right', frameon=False)

sns.despine()
plt.savefig('c18h12_hf.pdf', bbox_inches = 'tight', dpi=1000)

