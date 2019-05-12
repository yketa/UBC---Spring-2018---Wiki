"""
This script compares longitudinal and transversal collective mean square
displacements at their lag times of maximum susceptibility for different
rotation diffusion constants. (see
https://yketa.github.io/UBC_2018_Wiki/#CMSD%20at%20maximum%20susceptibility)

Files Cll_*.pickle contain longitudinal collective mean square displacements at
the lag times of maximum susceptibility for the corresponding rotation
diffusion constants.
Files Ctt_*.pickle contain transversal collective mean square displacements at
the lag times of maximum susceptibility for the corresponding rotation
diffusion constants.

Execution of this script requires the active_particles package (see
https://github.com/yketa/active_particles).
"""

import os

import pickle

import numpy as np

from active_particles.exponents import float_to_letters

from active_particles.plot.mpl_tools import FittingLine
from active_particles.plot.plot import list_colormap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# SCRIPT

if __name__ == '__main__':

    dir = os.path.dirname(os.path.realpath(__file__))   # directory containing this script

    # VARIABLES

    av_p_sep = 1.994824956985917    # average particle separation
    dL = 0.6324555320336759         # grid spacing in units of average particle separation

    dr_list = [2e-5, 7e-5, 2e-4, 7e-4, 2e-3, 7e-3]  # list of rotation diffusion constant to be compared
    Cll, Ctt = {}, {}                               # hash tables of cylindrical averages of mean square norms of dot and cross products of normalised wave vectors with displacement grids Fourier transform with rotation diffusion constants as keys
    for dr in dr_list:
        with open(os.path.join(dir, 'Cll_R%s.pickle' % float_to_letters(dr)),
            'rb') as Cll_file:
            _, _, Cll[dr] = pickle.load(Cll_file)
        with open(os.path.join(dir, 'Ctt_R%s.pickle' % float_to_letters(dr)),
            'rb') as Ctt_file:
            _, _, Ctt[dr] = pickle.load(Ctt_file)

    # PLOT

    mpl.rcParams.update({'font.size': 12})

    colors = list_colormap(dr_list) # hash table of line colors with rotation diffusion constants as keys

    fig = plt.figure()
    fig.set_size_inches(16, 16)
    fig.subplots_adjust(wspace=0.35)
    fig.suptitle(r'$\phi = 0.80, \tilde{v} = 1\cdot10^{-2}, N=1\cdot10^5$'
        + '\n' + r'$dL/a = %.2e$' % dL)

    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.25])

    ax_cll = plt.subplot(gs[0])
    ax_cll.set_xlabel(r'$\lambda/a = 2\pi/ka$')
    ax_cll.set_ylabel(r'$k^2 C^{||}(k, \Delta t^*(\tilde{\nu}_r))$')

    ax_ctt = plt.subplot(gs[1])
    ax_ctt.set_xlabel(r'$\lambda/a = 2\pi/ka$')
    ax_ctt.set_ylabel(r'$k^2 C^{\perp}(k, \Delta t^*(\tilde{\nu}_r))$')

    leg = plt.subplot(gs[2])
    leg.axis('off')
    leg.legend(handles=[Line2D([0], [0], color=colors[dr],
            label=r'$\tilde{\nu}_r=%.2e$' % dr)
            for dr in dr_list],
        loc='center')

    for dr in dr_list:
        ax_cll.loglog(
            2*np.pi/(Cll[dr][1:, 0]*av_p_sep),
            Cll[dr][1:, 1]*(Cll[dr][1:, 0]**2)
                # /np.max(Cll[dr][1:, 1]*(Cll[dr][1:, 0]**2))
                ,
            color=colors[dr], zorder=-dr_list.index(dr))
        ax_ctt.loglog(
            2*np.pi/(Ctt[dr][1:, 0]*av_p_sep),
            Ctt[dr][1:, 1]*(Ctt[dr][1:, 0]**2)
                # /np.max(Ctt[dr][1:, 1]*(Ctt[dr][1:, 0]**2))
                ,
            color=colors[dr], zorder=-dr_list.index(dr))

    fl_cll = FittingLine(ax_cll, 0,
        x_fit='(\lambda/a)', y_fit='k^2 C^{||}(k)',
        slider=False, legend=False)
    fl_ctt = FittingLine(ax_ctt, 0,
        x_fit='(\lambda/a)', y_fit='k^2 C^{\perp}(k)',
        slider=False, legend=False)

    plt.show()
