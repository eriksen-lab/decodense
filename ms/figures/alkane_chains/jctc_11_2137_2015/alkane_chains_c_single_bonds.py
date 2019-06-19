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

e_c6h14 = np.array([-15.683,-15.339,-14.136])
e_c7h16 = np.array([-16.591,-16.056,-14.725])
e_c8h18 = np.array([-17.500,-17.308,-16.645,-15.226])
e_c9h20 = np.array([-18.216,-17.897,-17.146,-15.661])
e_c10h22 = np.array([-18.933,-18.806,-18.399,-17.582,-16.047])
e_c11h24 = np.array([-19.523,-19.307,-18.834,-17.967,-16.392])
e_c12h26 = np.array([-20.112,-20.024,-19.742,-19.219,-18.312,-16.705])
e_c13h28 = np.array([-20.613,-20.459,-20.128,-19.565,-18.625,-16.991])

# bond types

type_alpha = np.array([e_c6h14[-1], e_c7h16[-1], e_c8h18[-1], e_c9h20[-1], e_c10h22[-1], e_c11h24[-1], e_c12h26[-1], e_c13h28[-1]])
atoms_alpha = np.arange(6,14)
type_beta = np.array([e_c6h14[-2], e_c7h16[-2], e_c8h18[-2], e_c9h20[-2], e_c10h22[-2], e_c11h24[-2], e_c12h26[-2], e_c13h28[-2]])
atoms_beta = np.arange(6,14)
type_gamma = np.array([e_c6h14[-3], e_c7h16[-3], e_c8h18[-3], e_c9h20[-3], e_c10h22[-3], e_c11h24[-3], e_c12h26[-3], e_c13h28[-3]])
atoms_gamma = np.arange(6,14)
type_delta = np.array([e_c8h18[-4], e_c9h20[-4], e_c10h22[-4], e_c11h24[-4], e_c12h26[-4], e_c13h28[-4]])
atoms_delta = np.arange(8,14)
type_epsilon = np.array([e_c10h22[-5], e_c11h24[-5], e_c12h26[-5], e_c13h28[-5]])
atoms_epsilon = np.arange(10,14)
type_zeta = np.array([e_c12h26[-6], e_c13h28[-6]])
atoms_zeta = np.arange(12,14)

# plotting domains

domain_alpha = np.linspace(atoms_alpha[0], atoms_alpha[-1], \
                           num=(atoms_alpha[-1] - atoms_alpha[0]) * 100)
domain_beta = np.linspace(atoms_beta[0], atoms_beta[-1], \
                           num=(atoms_beta[-1] - atoms_beta[0]) * 100)
domain_gamma = np.linspace(atoms_gamma[0], atoms_gamma[-1], \
                           num=(atoms_gamma[-1] - atoms_gamma[0]) * 100)
domain_delta = np.linspace(atoms_delta[0], atoms_delta[-1], \
                           num=(atoms_delta[-1] - atoms_delta[0]) * 100)
domain_epsilon = np.linspace(atoms_epsilon[0], atoms_epsilon[-1], \
                           num=(atoms_epsilon[-1] - atoms_epsilon[0]) * 100)
domain_zeta = np.linspace(atoms_zeta[0], atoms_zeta[-1], \
                           num=(atoms_zeta[-1] - atoms_zeta[0]) * 100)

sns.set(style='darkgrid', font='DejaVu Sans')

fig, ax = plt.subplots()

palette = sns.color_palette('Set2')

opt, r_2 = fit(func, atoms_alpha, type_alpha)
ax.plot(domain_alpha, func(domain_alpha, *opt), linewidth=2, linestyle='-', color=palette[0], label='C-C [$\\alpha$] ($r^2 = {:.3f}$)'.format(r_2))
print('\n\nf[alpha]   = {:.2f} * N {:+2f} / N {:.2f} --- r_2 = {:.3f}'.format(*opt, r_2))
ax.scatter(atoms_alpha, type_alpha, s=150, marker='.', color=palette[0])
opt, r_2 = fit(func, atoms_beta, type_beta)
ax.plot(domain_beta, func(domain_beta, *opt), linewidth=2, linestyle='-', color=palette[2], label='C-C [$\\beta$] ($r^2 = {:.3f}$)'.format(r_2))
print('f[beta]    = {:.2f} * N {:+2f} / N {:.2f} --- r_2 = {:.3f}'.format(*opt, r_2))
ax.scatter(atoms_beta, type_beta, s=150, marker='.', color=palette[2])
opt, r_2 = fit(func, atoms_gamma, type_gamma)
ax.plot(domain_gamma, func(domain_gamma, *opt), linewidth=2, linestyle='-', color=palette[3], label='C-C [$\\gamma$] ($r^2 = {:.3f}$)'.format(r_2))
print('f[gamma]   = {:.2f} * N {:+2f} / N {:.2f} --- r_2 = {:.3f}'.format(*opt, r_2))
ax.scatter(atoms_gamma, type_gamma, s=150, marker='.', color=palette[3])
opt, r_2 = fit(func, atoms_delta, type_delta)
ax.plot(domain_delta, func(domain_delta, *opt), linewidth=2, linestyle='-', color=palette[4], label='C-C [$\\delta$] ($r^2 = {:.3f}$)'.format(r_2))
print('f[delta]   = {:.2f} * N {:+2f} / N {:.2f} --- r_2 = {:.3f}'.format(*opt, r_2))
ax.scatter(atoms_delta, type_delta, s=150, marker='.', color=palette[4])
opt, r_2 = fit(func, atoms_epsilon, type_epsilon)
ax.plot(domain_epsilon, func(domain_epsilon, *opt), linewidth=2, linestyle='-', color=palette[1], label='C-C [$\\epsilon$] ($r^2 = {:.3f}$)'.format(r_2))
print('f[epsilon] = {:.2f} * N {:+2f} / N {:.2f} --- r_2 = {:.3f}'.format(*opt, r_2))
ax.scatter(atoms_epsilon, type_epsilon, s=150, marker='.', color=palette[1])
#opt, r_2 = fit(func, atoms_zeta, type_zeta)
#ax.plot(domain_zeta, func(domain_zeta, *opt), linewidth=2, linestyle='-', color=palette[5], label='C-C [$\\zeta$] ($r^2 = {:.3f}$)'.format(r_2))
#print('f[zeta]    = {:.2f} * N {:+2f} / N {:.2f} --- r_2 = {:.3f}\n\n'.format(*opt, r_2))
ax.plot(atoms_zeta, type_zeta, linewidth=2, linestyle='-', color=palette[5], label='C-C [$\\zeta$] ($r^2 = {:.3f}$)'.format(1.))
print('f[zeta]    = None\n\n')
ax.scatter(atoms_zeta, type_zeta, s=150, marker='.', color=palette[5])

ax.xaxis.grid(False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xlabel('Number of Carbon Atoms')
ax.set_ylabel('Energy Contribution (in au)')
ax.legend(loc='lower left', frameon=False)

sns.despine()
plt.savefig('alkane_chains_c_single_bonds.pdf', bbox_inches = 'tight', dpi=1000)

