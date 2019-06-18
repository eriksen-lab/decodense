#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple, HandlerLine2D
import seaborn as sns

n_carbons = np.arange(6,12)

e_c6h14 = np.array([-15.683,-15.339,-14.136])
e_c7h16 = np.array([-16.591,-16.056,-14.725])
e_c8h18 = np.array([-17.500,-17.308,-16.645,-15.226])
e_c9h20 = np.array([-18.216,-17.897,-17.146,-15.661])
e_c10h22 = np.array([-18.933,-18.806,-18.399,-17.582,-16.047])
e_c11h24 = np.array([-19.523,-19.307,-18.834,-17.967,-16.392])

type_alpha = np.array([e_c6h14[-1], e_c7h16[-1], e_c8h18[-1], e_c9h20[-1], e_c10h22[-1], e_c11h24[-1]])
type_beta = np.array([e_c6h14[-2], e_c7h16[-2], e_c8h18[-2], e_c9h20[-2], e_c10h22[-2], e_c11h24[-2]])
type_gamma = np.array([e_c8h18[-3], e_c9h20[-3], e_c10h22[-3], e_c11h24[-3]])
type_delta = np.array([e_c10h22[-4], e_c11h24[-4]])
type_omega = np.array([e_c6h14[0], e_c7h16[0], e_c8h18[0], e_c9h20[0], e_c10h22[0], e_c11h24[0]])

sns.set(style='darkgrid', font='DejaVu Sans')

fig, ax = plt.subplots()

palette = sns.color_palette('Set2')

ax.plot(n_carbons, type_alpha, lw=2, ls='-', marker='.', ms=10, color=palette[0], label='C-C [$\\alpha$]')
ax.plot(n_carbons, type_beta, lw=2, ls='-', marker='.', ms=10, color=palette[2], label='C-C [$\\beta$]')
ax.plot(n_carbons[2:], type_gamma, lw=2, ls='-', marker='.', ms=10, color=palette[3], label='C-C [$\\gamma$]')
ax.plot(n_carbons[4:], type_delta, lw=2, ls='-', marker='.', ms=10, color=palette[4], label='C-C [$\\delta$]')
ax.plot(n_carbons, type_omega, lw=2, ls='-', marker='.', ms=10, color=palette[1], label='C-C [$\\omega$]')

ax.xaxis.grid(False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xlabel('Number of Carbon Atoms')
ax.set_ylabel('Energy Contribution (in au)')
ax.legend(loc='lower left', markerscale=0, frameon=False)

sns.despine()
plt.savefig('alkane_chains_c_single_bonds.pdf', bbox_inches = 'tight', dpi=1000)

