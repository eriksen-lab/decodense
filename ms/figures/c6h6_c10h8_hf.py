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

c6h6_hf_ibo = np.sort(np.array([
-38.913,
-15.116,
-27.921,
-11.908,
-38.913,
-27.921,
-11.908,
-38.913,
-15.116,
-11.908,
-38.913,
-27.921,
-11.908,
-38.913,
-15.116,
-11.908,
-38.913,
-11.908
]))

c10h8_hf_ibo = np.sort(np.array([
-42.716,
-34.760,
-19.713,
-15.487,
-41.885,
-18.043,
-14.529,
-41.885,
-34.760,
-14.529,
-42.716,
-19.713,
-15.487,
-44.686,
-38.698,
-19.713,
-44.686,
-19.713,
-42.716,
-34.760,
-15.487,
-41.885,
-18.043,
-14.529,
-41.885,
-34.760,
-14.529,
-42.716,
-15.487
]))

n_core_c6h6 = 6
n_c_c_double_c6h6 = 3
n_c_c_single_c6h6 = 3
n_c_h_c6h6 = 6
n_core_c10h8 = 10
n_c_c_double_c10h8 = 5
n_c_c_single_c10h8 = 6
n_c_h_c10h8 = 8

sns.set(style='darkgrid', font='DejaVu Sans')

fig, ax = plt.subplots()

palette = sns.color_palette('Set2')

c6h6_ibo_core = ax.scatter(np.arange(n_core_c6h6), \
                           c6h6_hf_ibo[:n_core_c6h6], \
                           s=30, marker='x', color=palette[3])
c6h6_ibo_core_fake = ax.scatter(n_core_c6h6, \
                                 c6h6_hf_ibo[n_core_c6h6], \
                                 s=30, marker='x', color=palette[3])
c6h6_ibo_c_c_double = ax.scatter(np.arange(n_core_c6h6, (n_core_c6h6 + n_c_c_double_c6h6)), \
                                  c6h6_hf_ibo[n_core_c6h6:(n_core_c6h6 + n_c_c_double_c6h6)], \
                                  s=30, marker='x', color=palette[0])
c6h6_ibo_c_c_single = ax.scatter(np.arange((n_core_c6h6 + n_c_c_double_c6h6), \
                                            (n_core_c6h6 + n_c_c_double_c6h6 + n_c_c_single_c6h6)), \
                                  c6h6_hf_ibo[(n_core_c6h6 + n_c_c_double_c6h6):(n_core_c6h6 + n_c_c_double_c6h6 + n_c_c_single_c6h6)], \
                                  s=30, marker='x', color=palette[1])
c6h6_ibo_c_h = ax.scatter(np.arange((n_core_c6h6 + n_c_c_double_c6h6 + n_c_c_single_c6h6), \
                                     (n_core_c6h6 + n_c_c_double_c6h6 + n_c_c_single_c6h6 + n_c_h_c6h6)), \
                           c6h6_hf_ibo[(n_core_c6h6 + n_c_c_double_c6h6 + n_c_c_single_c6h6):(n_core_c6h6 + n_c_c_double_c6h6 + n_c_c_single_c6h6 + n_c_h_c6h6)], \
                           s=30, marker='x', color=palette[2])
c10h8_ibo_core = ax.scatter(np.arange(n_core_c10h8), \
                            c10h8_hf_ibo[:n_core_c10h8], \
                            s=150, marker='.', color=palette[3])
c10h8_ibo_core_fake = ax.scatter(n_core_c10h8, \
                                  c10h8_hf_ibo[n_core_c10h8], \
                                  s=150, marker='.', color=palette[3])
c10h8_ibo_c_c_double = ax.scatter(np.arange(n_core_c10h8, (n_core_c10h8 + n_c_c_double_c10h8)), \
                                   c10h8_hf_ibo[n_core_c10h8:(n_core_c10h8 + n_c_c_double_c10h8)], \
                                   s=150, marker='.', color=palette[0])
c10h8_ibo_c_c_single = ax.scatter(np.arange((n_core_c10h8 + n_c_c_double_c10h8), \
                                             (n_core_c10h8 + n_c_c_double_c10h8 + n_c_c_single_c10h8)), \
                                   c10h8_hf_ibo[(n_core_c10h8 + n_c_c_double_c10h8):(n_core_c10h8 + n_c_c_double_c10h8 + n_c_c_single_c10h8)], \
                                   s=150, marker='.', color=palette[1])
c10h8_ibo_c_h = ax.scatter(np.arange((n_core_c10h8 + n_c_c_double_c10h8 + n_c_c_single_c10h8), \
                                      (n_core_c10h8 + n_c_c_double_c10h8 + n_c_c_single_c10h8 + n_c_h_c10h8)), \
                            c10h8_hf_ibo[(n_core_c10h8 + n_c_c_double_c10h8 + n_c_c_single_c10h8):(n_core_c10h8 + n_c_c_double_c10h8 + n_c_c_single_c10h8 + n_c_h_c10h8)], \
                            s=150, marker='.', color=palette[2])
ax.xaxis.grid(False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.set_xlabel('Contribution')
ax.set_ylabel('Energy Contribution (in au)')
ax.legend([(c6h6_ibo_core_fake, c6h6_ibo_c_c_double, c6h6_ibo_c_c_single, c6h6_ibo_c_h), \
            (c10h8_ibo_core_fake, c10h8_ibo_c_c_double, c10h8_ibo_c_c_single, c10h8_ibo_c_h)], \
            ['Benzene', 'Naphthalene'], loc='lower right', frameon=False, \
            numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})

sns.despine()
plt.savefig('c6h6_c10h8_hf.pdf', bbox_inches = 'tight', dpi=1000)

