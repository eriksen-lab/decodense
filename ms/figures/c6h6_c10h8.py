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
-38.913,
-38.913,
-38.913,
-38.913,
-38.913,
-38.913,
-27.920,
-27.920,
-27.920,
-15.115,
-15.115,
-15.115,
-11.909,
-11.909,
-11.909,
-11.909,
-11.909,
-11.909
])

c10h8_c_core = np.arange(10)
c10h8_c_double_c = np.arange(10,15)
c10h8_c_single_c = np.arange(15,21)
c10h8_c_h = np.arange(21,29)

c10h8_hf_ibo = np.array([
-44.686,
-44.686,
-42.716,
-42.716,
-42.716,
-42.716,
-41.885,
-41.885,
-41.885,
-41.885,
-38.572,
-34.944,
-34.944,
-34.625,
-34.625,
-19.715,
-19.715,
-19.715,
-19.715,
-18.036,
-18.036,
-15.490,
-15.490,
-15.490,
-15.490,
-14.534,
-14.534,
-14.534,
-14.534
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
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.set_xlabel('Contribution')
ax.set_ylabel('Energy Contribution (in au)')
ax.legend([(c6h6_ibo_c_core, c6h6_ibo_c_double_c, c6h6_ibo_c_single_c, c6h6_ibo_c_h), \
            (c10h8_ibo_c_core, c10h8_ibo_c_double_c, c10h8_ibo_c_single_c, c10h8_ibo_c_h)], \
            ['Benzene', 'Naphthalene'], loc='lower right', frameon=False, \
            numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})

sns.despine()
plt.savefig('c6h6_c10h8_hf.pdf', bbox_inches = 'tight', dpi=1000)

