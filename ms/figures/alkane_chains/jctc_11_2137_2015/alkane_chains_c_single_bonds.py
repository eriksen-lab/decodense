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

n_carbons = np.arange(6,12)

e_c6h14 = np.array([-15.683,-15.339,-14.136])
e_c7h16 = np.array([-16.591,-16.056,-14.725])
e_c8h18 = np.array([-17.500,-17.308,-16.645,-15.226])
e_c9h20 = np.array([-18.216,-17.897,-17.146,-15.661])
e_c10h22 = np.array([-18.933,-18.806,-18.399,-17.582,-16.047])
e_c11h24 = np.array([-19.523,-19.307,-18.834,-17.967,-16.392])

# bond types

type_alpha = np.array([e_c6h14[-1], e_c7h16[-1], e_c8h18[-1], e_c9h20[-1], e_c10h22[-1], e_c11h24[-1]])
type_beta = np.array([e_c6h14[-2], e_c7h16[-2], e_c8h18[-2], e_c9h20[-2], e_c10h22[-2], e_c11h24[-2]])
type_gamma = np.array([e_c8h18[-3], e_c9h20[-3], e_c10h22[-3], e_c11h24[-3]])
type_delta = np.array([e_c10h22[-4], e_c11h24[-4]])
type_omega = np.array([e_c6h14[0], e_c7h16[0], e_c8h18[0], e_c9h20[0], e_c10h22[0], e_c11h24[0]])

# plotting domains

domain_alpha = np.linspace(n_carbons[0], n_carbons[-1], num=n_carbons.size * 50)
domain_beta = np.linspace(n_carbons[0], n_carbons[-1], num=n_carbons.size * 50)
domain_gamma = np.linspace(n_carbons[2], n_carbons[-1], num=n_carbons[2:].size * 50)
domain_delta = np.linspace(n_carbons[4], n_carbons[-1], num=n_carbons[4:].size * 50)
domain_omega = np.linspace(n_carbons[0], n_carbons[-1], num=n_carbons.size * 50)

sns.set(style='darkgrid', font='DejaVu Sans')

fig, ax = plt.subplots()

palette = sns.color_palette('Set2')

opt, r_2 = fit(func, n_carbons, type_alpha)
ax.plot(domain_alpha, func(domain_alpha, *opt), linewidth=2, linestyle='-', color=palette[0], label='C-C [$\\alpha$] ($r^2 = {:.3f}$)'.format(r_2))
print('\n\nf[alpha] = {:.2f} * N {:+2f} / N {:.2f} --- r_2 = {:.3f}'.format(*opt, r_2))
ax.scatter(n_carbons, type_alpha, s=150, marker='.', color=palette[0])
opt, r_2 = fit(func, n_carbons, type_beta)
ax.plot(domain_beta, func(domain_beta, *opt), linewidth=2, linestyle='-', color=palette[2], label='C-C [$\\beta$] ($r^2 = {:.3f}$)'.format(r_2))
print('f[beta]  = {:.2f} * N {:+2f} / N {:.2f} --- r_2 = {:.3f}'.format(*opt, r_2))
ax.scatter(n_carbons, type_beta, s=150, marker='.', color=palette[2])
opt, r_2 = fit(func, n_carbons[2:], type_gamma)
ax.plot(domain_gamma, func(domain_gamma, *opt), linewidth=2, linestyle='-', color=palette[3], label='C-C [$\\gamma$] ($r^2 = {:.3f}$)'.format(r_2))
print('f[gamma] = {:.2f} * N {:+2f} / N {:.2f} --- r_2 = {:.3f}'.format(*opt, r_2))
ax.scatter(n_carbons[2:], type_gamma, s=150, marker='.', color=palette[3])
#opt, r_2 = fit(func, n_carbons[4:], type_delta)
#ax.plot(n_carbons[4:], func(n_carbons[4:], *opt), linewidth=2, linestyle='-', color=palette[4], label='C-C [$\\delta$] ($r^2 = {:.3f}$)'.format(r_2))
ax.plot(n_carbons[4:], type_delta, linewidth=2, linestyle='-', color=palette[4], label='C-C [$\\delta$] ($r^2 = {:.3f}$)'.format(1.))
print('f[delta] = None')
ax.scatter(n_carbons[4:], type_delta, s=150, marker='.', color=palette[4])
opt, r_2 = fit(func, n_carbons, type_omega)
ax.plot(domain_omega, func(domain_omega, *opt), linewidth=2, linestyle='-', color=palette[1], label='C-C [$\\omega$] ($r^2 = {:.3f}$)'.format(r_2))
print('f[omega] = {:.2f} * N {:+2f} / N {:.2f} --- r_2 = {:.3f}\n\n'.format(*opt, r_2))
ax.scatter(n_carbons, type_omega, s=150, marker='.', color=palette[1])

ax.xaxis.grid(False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xlabel('Number of Carbon Atoms')
ax.set_ylabel('Energy Contribution (in au)')
ax.legend(loc='lower left', frameon=False)

sns.despine()
plt.savefig('alkane_chains_c_single_bonds.pdf', bbox_inches = 'tight', dpi=1000)

