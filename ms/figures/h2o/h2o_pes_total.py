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

e_hf = np.array([
-75.35661,
-75.60638,
-75.77584,
-75.88812,
-75.95940,
-76.00119,
-76.02174,
-76.02702,
-76.02144,
-76.00823,
-75.98975,
-75.96775,
-75.94346,
-75.91782,
-75.89147,
-75.83843,
-75.78675,
-75.73762,
-75.69155,
-75.64867,
-75.60894,
-75.57225,
#-75.53909,
-75.50720,
#-75.47856,
-75.45221,
#-75.42799,
-75.40574,
#-75.38531,
-75.36658,
#-75.34942,
-75.33371
])

e_dft = np.array([
-75.63289,
-75.88554,
-76.05840,
-76.17451,
-76.25005,
-76.29650,
-76.32211,
-76.33284,
-76.33304,
-76.32589,
-76.31370,
-76.29814,
-76.28042,
-76.26138,
-76.24166,
-76.20180,
-76.16308,
-76.12660,
-76.09287,
-76.06201,
-76.03400,
-76.00875,
#-75.98609,
-75.96587,
#-75.94790,
-75.93196,
#-75.91787,
-75.90542,
#-75.89443,
-75.88476,
#-75.87625,
-75.86879
])

nuc = np.array([
+14.670,
+13.542,
+12.574,
+11.736,
+11.003,
+10.355,
+9.780,
+9.265,
+8.802,
+8.383,
+8.002,
+7.654,
+7.335,
+7.042,
+6.771,
+6.287,
+5.868,
+5.501,
+5.178,
+4.890,
+4.633,
+4.401,
#+4.191,
+4.001,
#+3.827,
+3.668,
#+3.521,
+3.385,
#+3.260,
+3.144,
#+3.035,
+2.934
])

xc = np.array([
-9.779,
-9.685,
-9.600,
-9.524,
-9.455,
-9.393,
-9.338,
-9.287,
-9.241,
-9.200,
-9.162,
-9.127,
-9.095,
-9.066,
-9.040,
-8.993,
-8.955,
-8.923,
-8.897,
-8.876,
-8.858,
-8.845,
#-8.834,
-8.826,
#-8.820,
-8.816,
#-8.813,
-8.812,
#-8.811,
-8.811,
#-8.812,
-8.814
])

sns.set(style='darkgrid', font='DejaVu Sans')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

palette = sns.color_palette('Set2')

ax1.plot(b, e_hf, linewidth=2, linestyle='-', color=palette[0], label='HF')
ax1.plot(b, e_dft, linewidth=2, linestyle='-', color=palette[1], label='PBE')
ax1.xaxis.grid(False)
ax1.legend(loc='lower right', frameon=False)
ax1.set_ylim(bottom=-76.6)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%6.1f'))
ax2.plot(b, nuc, linewidth=2, linestyle='-', color=palette[2], label='Nucl. Rep.')
ax2.xaxis.grid(False)
ax2.legend(loc='upper right', frameon=False)
ax2.set_ylim(top=16.0)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%6.1f'))
ax2.set_ylabel('Total Energy (in au)')
ax3.plot(b, xc, linewidth=2, linestyle='-', color=palette[3], label='XC Energy')
ax3.xaxis.grid(False)
ax3.legend(loc='lower right', frameon=False)
ax3.yaxis.set_major_formatter(FormatStrFormatter('%6.1f'))
ax3.set_xlabel('O-H Bond Length (in Å)')

fig.subplots_adjust(hspace=0.05)
sns.despine()
plt.savefig('h2o_pes_total.pdf', bbox_inches = 'tight', dpi=1000)

