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

# fitting functions

def func(x, a, b, c):
	return a * x + b / x + c

# fitting kernel

def fit(func, x, y):
	opt = curve_fit(func, x, y)[0]
	ss_res = np.sum((y - func(x, *opt)) ** 2)
	ss_tot = np.sum((y - np.mean(y)) ** 2)
	r_2 = 1. - (ss_res / ss_tot)
	return opt, r_2

# data

n_carbons = np.arange(6,17,2)

e_c6h8 = np.array([-27.861,-25.182])
e_c8h10 = np.array([-30.779,-27.149])
e_c10h12 = np.array([-33.693,-32.742,-28.629])
e_c12h14 = np.array([-35.655,-34.220,-29.816])
e_c14h16 = np.array([-37.616,-37.133,-35.405,-30.806])
e_c16h18 = np.array([-39.093,-38.318,-36.395,-31.656])

# bond types

type_alpha = np.array([e_c6h8[-1], e_c8h10[-1], e_c10h12[-1], e_c12h14[-1], e_c14h16[-1], e_c16h18[-1]])
type_beta = np.array([e_c10h12[-2], e_c12h14[-2], e_c14h16[-2], e_c16h18[-2]])
type_gamma = np.array([e_c14h16[-3], e_c16h18[-3]])
type_omega = np.array([e_c6h8[0], e_c8h10[0], e_c10h12[0], e_c12h14[0], e_c14h16[0], e_c16h18[0]])

# plotting domains

domain_alpha = np.linspace(n_carbons[0], n_carbons[-1], num=n_carbons.size * 50)
domain_beta = np.linspace(n_carbons[2], n_carbons[-1], num=n_carbons[2:].size * 50)
domain_gamma = np.linspace(n_carbons[4], n_carbons[-1], num=n_carbons[4:].size * 50)
domain_omega = np.linspace(n_carbons[0], n_carbons[-1], num=n_carbons.size * 50)

sns.set(style='darkgrid', font='DejaVu Sans')

fig, ax = plt.subplots()

palette = sns.color_palette('Set2')

opt, r_2 = fit(func, n_carbons, type_alpha)
ax.plot(domain_alpha, func(domain_alpha, *opt), linewidth=2, linestyle='-', color=palette[0], label='C=C [$\\alpha$] ($r^2 = {:.3f}$)'.format(r_2))
print('\n\nf[alpha] = {:.2f} * N {:+2f} / N {:.2f} --- r_2 = {:.3f}'.format(*opt, r_2))
ax.scatter(n_carbons, type_alpha, s=150, marker='.', color=palette[0])
opt, r_2 = fit(func, n_carbons[2:], type_beta)
ax.plot(domain_beta, func(domain_beta, *opt), linewidth=2, linestyle='-', color=palette[2], label='C=C [$\\beta$] ($r^2 = {:.3f}$)'.format(r_2))
print('f[beta]  = {:.2f} * N {:+2f} / N {:.2f} --- r_2 = {:.3f}'.format(*opt, r_2))
ax.scatter(n_carbons[2:], type_beta, s=150, marker='.', color=palette[2])
#opt, r_2 = fit(func, n_carbons[4:], type_gamma)
#ax.plot(n_carbons[4:], func(n_carbons[4:], *opt), linewidth=2, linestyle='-', color=palette[3], label='C-C [$\\gamma$] ($r^2 = {:.3f}$)'.format(r_2))
ax.plot(n_carbons[4:], type_gamma, linewidth=2, linestyle='-', color=palette[3], label='C=C [$\\gamma$] ($r^2 = {:.3f}$)'.format(1.))
print('f[gamma] = None')
ax.scatter(n_carbons[4:], type_gamma, s=150, marker='.', color=palette[3])
opt, r_2 = fit(func, n_carbons, type_omega)
ax.plot(domain_omega, func(domain_omega, *opt), linewidth=2, linestyle='-', color=palette[1], label='C=C [$\\omega$] ($r^2 = {:.3f}$)'.format(r_2))
print('f[omega] = {:.2f} * N {:+2f} / N {:.2f} --- r_2 = {:.3f}\n\n'.format(*opt, r_2))
ax.scatter(n_carbons, type_omega, s=150, marker='.', color=palette[1])

ax.xaxis.grid(False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xlabel('Number of Carbon Atoms')
ax.set_ylabel('Energy Contribution (in au)')
ax.legend(loc='lower left', markerscale=0, frameon=False)

sns.despine()
plt.savefig('polyacetylenes_c_double_bonds.pdf', bbox_inches = 'tight', dpi=1000)

