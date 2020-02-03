#!/usr/bin/env python

import copy
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple, HandlerLine2D
import seaborn as sns

# data

b = np.array([
0.60,
0.65,
0.70,
0.75,
0.80,
0.85,
0.90,
0.95,
1.00,
1.05,
1.10,
1.15,
1.20,
1.25,
1.30,
1.40,
1.50,
1.60,
1.70,
1.80,
1.90,
2.00,
#2.10,
2.20,
#2.30,
2.40,
#2.50,
2.60,
#2.70,
2.80,
#2.90,
3.00
])

e_hf_core = np.array([
-70.548,
-70.179,
-69.894,
-69.674,
-69.508,
-69.385,
-69.297,
-69.236,
-69.198,
-69.178,
-69.171,
-69.173,
-69.182,
-69.193,
-69.206,
-69.228,
-69.240,
-69.240,
-69.228,
-69.206,
-69.178,
-69.146,
#-69.143,
-69.055,
#-69.015,
-68.973,
#-68.931,
-68.889,
#-68.848,
-68.808,
#-68.769,
-68.731
])

e_hf_bond = np.array([
-9.740,
-9.485,
-9.228,
-8.975,
-8.727,
-8.486,
-8.253,
-8.028,
-7.813,
-7.607,
-7.410,
-7.224,
-7.048,
-6.883,
-6.728,
-6.449,
-6.207,
-6.000,
-5.821,
-5.666,
-5.532,
-5.413,
#-5.294,
-5.227,
#-5.145,
-5.073,
#-5.009,
-4.951,
#-4.899,
-4.851,
#-4.808,
-4.768
])

e_dft_core = np.array([
-63.585,
-63.235,
-62.960,
-62.745,
-62.578,
-62.452,
-62.358,
-62.291,
-62.247,
-62.219,
-62.203,
-62.196,
-62.193,
-62.191,
-62.189,
-62.180,
-62.159,
-62.127,
-62.087,
-62.040,
-61.990,
-61.939,
#-61.886,
-61.835,
#-61.784,
-61.735,
#-61.688,
-61.643,
#-61.599,
-61.557,
#-61.518,
-61.480
])

e_dft_bond = np.array([
-8.470,
-8.254,
-8.036,
-7.821,
-7.610,
-7.403,
-7.203,
-7.010,
-6.824,
-6.645,
-6.475,
-6.315,
-6.164,
-6.023,
-5.892,
-5.658,
-5.459,
-5.289,
-5.144,
-5.018,
-4.909,
-4.813,
#-4.729,
-4.653,
#-4.585,
-4.524,
#-4.469,
-4.418,
#-4.372,
-4.330,
#-4.291,
-4.255
])

sns.set(style='darkgrid', font='DejaVu Sans')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

palette = sns.color_palette('Set2')

ax1.plot(b, e_dft_bond, linewidth=2, linestyle='-', color=palette[1], label='PBE: O-H')
ax1.plot(b, e_hf_bond, linewidth=2, linestyle='-', color=palette[0], label='HF: O-H')
ax1.xaxis.grid(False)
ax1.legend(loc='center right', frameon=False)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%4.1f'))
ax1.set_ylim(bottom=-10.4,top=-3.6)
ax2.plot(b, e_dft_core, linewidth=2, linestyle='-', color=palette[1], label='PBE: O(1s)')
ax2.plot(b, e_hf_core, linewidth=2, linestyle='-', color=palette[0], label='HF: O(1s)')
ax2.xaxis.grid(False)
ax2.legend(loc='center right', frameon=False)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%4.1f'))
ax2.set_xlabel('O-H Bond Length (in Å)')

fig.text(0.02, 0.5, 'Energy Contribution (in au)', ha='center', va='center', rotation='vertical')

fig.subplots_adjust(hspace=0.05)
sns.despine()
plt.savefig('h2o_pes.pdf', bbox_inches = 'tight', dpi=1000)

