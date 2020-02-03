#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import seaborn as sns

c6h6_c_core = np.arange(6)
c6h6_c_double_c = np.arange(6,9)
c6h6_c_single_c = np.arange(9,12)
c6h6_c_h = np.arange(12,18)

c6h6_hf_ibo = np.array([
-38.915,
-38.915,
-38.915,
-38.915,
-38.915,
-38.915,
-27.950,
-27.950,
-27.950,
-15.136,
-15.136,
-15.136,
-11.898,
-11.898,
-11.898,
-11.898,
-11.898,
-11.898
])

c10h8_c_core = np.arange(10)
c10h8_c_double_c = np.arange(10,15)
c10h8_c_single_c = np.arange(15,21)
c10h8_c_h = np.arange(21,29)

c10h8_hf_ibo = np.array([
-44.685,
-44.685,
-42.718,
-42.718,
-42.718,
-42.718,
-41.887,
-41.887,
-41.887,
-41.887,
-38.704,
-34.786,
-34.786,
-34.786,
-34.786,
-19.733,
-19.733,
-19.733,
-19.733,
-18.062,
-18.062,
-15.477,
-15.477,
-15.477,
-15.477,
-14.517,
-14.517,
-14.517,
-14.517
])

sns.set(style='darkgrid', font='DejaVu Sans')

fig, ax = plt.subplots()

palette = sns.color_palette('Set2')

c6h6_ibo_c_core = ax.scatter(c6h6_c_core, c6h6_hf_ibo[c6h6_c_core], \
                             s=30, marker='x', color=palette[3])
c6h6_ibo_c_double_c = ax.scatter(c6h6_c_double_c, c6h6_hf_ibo[c6h6_c_double_c], \
                                 s=30, marker='x', color=palette[0])
c6h6_ibo_c_single_c = ax.scatter(c6h6_c_single_c, c6h6_hf_ibo[c6h6_c_single_c], \
                                 s=30, marker='x', color=palette[1])
c6h6_ibo_c_h = ax.scatter(c6h6_c_h, c6h6_hf_ibo[c6h6_c_h], \
                          s=30, marker='x', color=palette[2])

c10h8_ibo_c_core = ax.scatter(c10h8_c_core, c10h8_hf_ibo[c10h8_c_core], \
                              s=150, marker='.', color=palette[3])
c10h8_ibo_c_double_c = ax.scatter(c10h8_c_double_c, c10h8_hf_ibo[c10h8_c_double_c], \
                                  s=150, marker='.', color=palette[0])
c10h8_ibo_c_single_c = ax.scatter(c10h8_c_single_c, c10h8_hf_ibo[c10h8_c_single_c], \
                                  s=150, marker='.', color=palette[1])
c10h8_ibo_c_h = ax.scatter(c10h8_c_h, c10h8_hf_ibo[c10h8_c_h], \
                           s=150, marker='.', color=palette[2])

ax.xaxis.grid(False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.set_ylim(top=-8.0)
ax.set_xlabel('orb-RDM1')
#ax.set_ylabel('Energy Contribution (in au)')
ax.legend([(c6h6_ibo_c_core, c6h6_ibo_c_double_c, c6h6_ibo_c_single_c, c6h6_ibo_c_h), \
            (c10h8_ibo_c_core, c10h8_ibo_c_double_c, c10h8_ibo_c_single_c, c10h8_ibo_c_h)], \
            ['Benzene', 'Naphthalene'], loc='lower right', frameon=False, \
            numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})

sns.despine()
plt.savefig('c6h6_c10h8_hf.pdf', bbox_inches = 'tight', dpi=1000)

