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
# 631g
#-50.523,
#-51.339,
#-27.502,
#-22.962,
#-51.054,
#-27.116,
#-25.434,
#-51.054,
#-51.229,
#-25.434,
#-50.523,
#-27.502,
#-22.962,
#-52.559,
#-28.631,
#-53.458,
#-52.559,
#-53.161,
#-51.194,
#-27.869,
#-23.589,
#-52.559,
#-28.631,
#-27.502,
#-52.559,
#-53.458,
#-27.502,
#-51.194,
#-23.589,
#-47.935,
#-44.684,
#-20.614,
#-46.466,
#-22.580,
#-18.803,
#-46.466,
#-44.511,
#-18.803,
#-47.935,
#-20.614,
#-50.523,
#-51.339,
#-22.962,
#-51.054,
#-27.116,
#-25.434,
#-51.054,
#-51.229,
#-25.434,
#-50.523,
#-22.962,
#-47.935,
#-44.684,
#-20.614,
#-46.466,
#-22.580,
#-18.803,
#-46.466,
#-44.511,
#-18.803,
#-47.935,
#-20.614
#ccpvdz
-50.525,
-51.345,
-27.517,
-22.955,
-51.054,
-27.141,
-25.453,
-51.054,
-51.238,
-25.453,
-50.525,
-27.517,
-22.955,
-52.559,
-28.657,
-53.456,
-52.559,
-53.174,
-51.196,
-27.883,
-23.582,
-52.559,
-28.657,
-27.517,
-52.559,
-53.456,
-27.517,
-51.196,
-23.582,
-47.937,
-44.700,
-20.608,
-46.469,
-22.603,
-18.790,
-46.469,
-44.549,
-18.790,
-47.937,
-20.608,
-50.525,
-51.345,
-22.955,
-51.054,
-27.141,
-25.453,
-51.054,
-51.238,
-25.453,
-50.525,
-22.955,
-47.937,
-44.700,
-20.608,
-46.469,
-22.603,
-18.790,
-46.469,
-44.549,
-18.790,
-47.937,
-20.608
]))

print('{:}'.format(hf_ibo))

n_rings = 5

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
plt.savefig('c22h14_hf.pdf', bbox_inches = 'tight', dpi=1000)

