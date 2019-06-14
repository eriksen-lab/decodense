#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

c_double_c = np.array(list(range(0,3))+list(range(7,11))+list(range(29,33)))
c_core = np.array(list(range(3,7))+list(range(11,29)))
c_single_c = np.array(list(range(33,46))+list(range(52,54)))
c_h = np.array(list(range(46,52))+list(range(54,62)))

res = np.array([
-53.458,
-53.458,
-53.161,
-52.559,
-52.559,
-52.559,
-52.559,
-51.339,
-51.339,
-51.229,
-51.229,
-51.194,
-51.194,
-51.054,
-51.054,
-51.054,
-51.054,
-50.523,
-50.523,
-50.523,
-50.523,
-47.935,
-47.935,
-47.935,
-47.935,
-46.466,
-46.466,
-46.466,
-46.466,
-44.684,
-44.684,
-44.511,
-44.511,
-28.631,
-28.631,
-27.869,
-27.502,
-27.502,
-27.502,
-27.502,
-27.116,
-27.116,
-25.434,
-25.434,
-25.434,
-25.434,
-23.589,
-23.589,
-22.962,
-22.962,
-22.962,
-22.962,
-22.580,
-22.580,
-20.614,
-20.614,
-20.614,
-20.614,
-18.803,
-18.803,
-18.803,
-18.803
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
plt.savefig('c22h14_hf.pdf', bbox_inches = 'tight', dpi=1000)

