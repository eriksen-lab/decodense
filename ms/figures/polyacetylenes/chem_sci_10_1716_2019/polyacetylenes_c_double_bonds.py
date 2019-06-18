#!/usr/bin/env python

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple, HandlerLine2D
import seaborn as sns

n_carbons = np.arange(6,17,2)

e_c6h8 = np.array([-27.861,-25.182])
e_c8h10 = np.array([-30.779,-27.149])
e_c10h12 = np.array([-33.693,-32.742,-28.629])
e_c12h14 = np.array([-35.655,-34.220,-29.816])
e_c14h16 = np.array([-37.616,-37.133,-35.405,-30.806])
e_c16h18 = np.array([-39.093,-38.318,-36.395,-31.656])

type_alpha = np.array([e_c6h8[-1], e_c8h10[-1], e_c10h12[-1], e_c12h14[-1], e_c14h16[-1], e_c16h18[-1]])
type_beta = np.array([e_c10h12[-2], e_c12h14[-2], e_c14h16[-2], e_c16h18[-2]])
type_gamma = np.array([e_c14h16[-3], e_c16h18[-3]])
type_omega = np.array([e_c6h8[0], e_c8h10[0], e_c10h12[0], e_c12h14[0], e_c14h16[0], e_c16h18[0]])

sns.set(style='darkgrid', font='DejaVu Sans')

fig, ax = plt.subplots()

palette = sns.color_palette('Set2')

ax.plot(n_carbons, type_alpha, lw=2, ls='-', marker='.', ms=10, color=palette[0], label='C=C [$\\alpha$]')
ax.plot(n_carbons[2:], type_beta, lw=2, ls='-', marker='.', ms=10, color=palette[2], label='C=C [$\\beta$]')
ax.plot(n_carbons[4:], type_gamma, lw=2, ls='-', marker='.', ms=10, color=palette[3], label='C=C [$\\gamma$]')
ax.plot(n_carbons, type_omega, lw=2, ls='-', marker='.', ms=10, color=palette[1], label='C=C [$\\omega$]')

ax.xaxis.grid(False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xlabel('Number of Carbon Atoms')
ax.set_ylabel('Energy Contribution (in au)')
ax.legend(loc='lower left', markerscale=0, frameon=False)

sns.despine()
plt.savefig('polyacetylenes_c_double_bonds.pdf', bbox_inches = 'tight', dpi=1000)

