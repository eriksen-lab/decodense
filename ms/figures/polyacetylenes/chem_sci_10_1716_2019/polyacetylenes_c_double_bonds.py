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

e_c6h8 = np.array([-27.861,-25.182])
e_c8h10 = np.array([-30.779,-27.149])
e_c10h12 = np.array([-33.693,-32.742,-28.629])
e_c12h14 = np.array([-35.655,-34.220,-29.816])
e_c14h16 = np.array([-37.616,-37.133,-35.405,-30.806])
e_c16h18 = np.array([-39.093,-38.318,-36.395,-31.656])

# bond types

type_alpha = np.array([e_c6h8[-1], e_c8h10[-1], e_c10h12[-1], e_c12h14[-1], e_c14h16[-1], e_c16h18[-1]])
atoms_alpha = np.arange(6,17,2)
type_beta = np.array([e_c6h8[-2], e_c8h10[-2], e_c10h12[-2], e_c12h14[-2], e_c14h16[-2], e_c16h18[-2]])
atoms_beta = np.arange(6,17,2)
type_gamma = np.array([e_c10h12[-3], e_c12h14[-3], e_c14h16[-3], e_c16h18[-3]])
atoms_gamma = np.arange(10,17,2)
type_delta = np.array([e_c14h16[-4], e_c16h18[-4]])
atoms_delta = np.arange(14,17,2)

# plotting domains

domain_alpha = np.linspace(atoms_alpha[0], atoms_alpha[-1], \
                           num=(atoms_alpha[-1] - atoms_alpha[0]) * 100)
domain_beta = np.linspace(atoms_beta[0], atoms_beta[-1], \
                           num=(atoms_beta[-1] - atoms_beta[0]) * 100)
domain_gamma = np.linspace(atoms_gamma[0], atoms_gamma[-1], \
                           num=(atoms_gamma[-1] - atoms_gamma[0]) * 100)
domain_delta = np.linspace(atoms_delta[0], atoms_delta[-1], \
                           num=(atoms_delta[-1] - atoms_delta[0]) * 100)

sns.set(style='darkgrid', font='DejaVu Sans')

fig, ax = plt.subplots()

palette = sns.color_palette('Set2')

opt, r_2 = fit(func, atoms_alpha, type_alpha)
ax.plot(domain_alpha, func(domain_alpha, *opt), linewidth=2, linestyle='-', color=palette[0], label='C=C [$\\alpha$] ($r^2 = {:.3f}$)'.format(r_2))
print('\n\nf[alpha] = {:.2f} * N {:+2f} / N {:.2f} --- r_2 = {:.3f}'.format(*opt, r_2))
ax.scatter(atoms_alpha, type_alpha, s=150, marker='.', color=palette[0])
opt, r_2 = fit(func, atoms_beta, type_beta)
ax.plot(domain_beta, func(domain_beta, *opt), linewidth=2, linestyle='-', color=palette[2], label='C=C [$\\beta$] ($r^2 = {:.3f}$)'.format(r_2))
print('f[beta]  = {:.2f} * N {:+2f} / N {:.2f} --- r_2 = {:.3f}'.format(*opt, r_2))
ax.scatter(atoms_beta, type_beta, s=150, marker='.', color=palette[2])
opt, r_2 = fit(func, atoms_gamma, type_gamma)
ax.plot(domain_gamma, func(domain_gamma, *opt), linewidth=2, linestyle='-', color=palette[3], label='C=C [$\\gamma$] ($r^2 = {:.3f}$)'.format(r_2))
print('f[gamma] = {:.2f} * N {:+2f} / N {:.2f} --- r_2 = {:.3f}'.format(*opt, r_2))
ax.scatter(atoms_gamma, type_gamma, s=150, marker='.', color=palette[3])
#opt, r_2 = fit(func, atoms_delta, type_delta)
#ax.plot(domain_delta, func(domain_delta, *opt), linewidth=2, linestyle='-', color=palette[1], label='C=C [$\\delta$] ($r^2 = {:.3f}$)'.format(r_2))
#print('f[delta] = {:.2f} * N {:+2f} / N {:.2f} --- r_2 = {:.3f}\n\n'.format(*opt, r_2))
ax.plot(atoms_delta, type_delta, linewidth=2, linestyle='-', color=palette[4], label='C=C [$\\delta$]  ($r^2 = {:.3f}$)'.format(1.))
print('f[delta] = None\n\n')
ax.scatter(atoms_delta, type_delta, s=150, marker='.', color=palette[4])

ax.xaxis.grid(False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xlabel('Number of Carbon Atoms')
ax.set_ylabel('Energy Contribution (in au)')
ax.legend(loc='lower left', markerscale=0, frameon=False)

sns.despine()
plt.savefig('polyacetylenes_c_double_bonds.pdf', bbox_inches = 'tight', dpi=1000)

