"""
This script plots local packing fraction distributions at (phi = 0.80, vzero =
1e-2, dr = 2e-5) and (phi = 0.80, vzero = 1e-1, dr = 5e-4). (see
https://yketa.github.io/UBC_2018_Wiki/#Strain%20correlations%20at%20high%20activity)

File varN_Vj1000_Rg2000.pickle contains a list of densities at (phi = 0.80,
vzero = 1e-2, dr = 2e-5) and file varN_Vk1000_Rh5000.pickle contains a list of
densities at (phi = 0.80, vzero = 1e-1, dr = 5e-4).

Execution of this script requires the active_particles package (see
https://github.com/yketa/active_particles).
"""

import os

import pickle

import numpy as np

from active_particles.init import set_env
set_env('SHOW', True)

from active_particles.analysis.varn import histogram, Plot

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if __name__ == '__main__':

    # VARIABLES

    Nbins = 100     # number of bins for the histogram
    phimax = 1.2    # maximum local density for histogram

    # PLOT

    mpl.rcParams.update({'font.size': 15})

    plot = Plot(suptitle=False)

    with open(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
            'varN_Vj1000_Rg2000.pickle'),
        'rb') as varN_file:
        densities_Vj1000_Rg2000 = pickle.load(varN_file)
    bins_Vj1000_Rg2000, hist_Vj1000_Rg2000 = histogram(
        densities_Vj1000_Rg2000, Nbins, phimax)
    philocmax_Vj1000_Rg2000, Pphilocmax_Vj1000_Rg2000 = (
        bins_Vj1000_Rg2000[np.argmax(hist_Vj1000_Rg2000)],
        np.max(hist_Vj1000_Rg2000))
    line_Vj1000_Rg2000 = plot.add_hist(
        bins_Vj1000_Rg2000, hist_Vj1000_Rg2000, peak=True)

    with open(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
            'varN_Vk1000_Rh5000.pickle'),
        'rb') as varN_file:
        densities_Vk1000_Rh5000 = pickle.load(varN_file)
    bins_Vk1000_Rh5000, hist_Vk1000_Rh5000 = histogram(
        densities_Vk1000_Rh5000, Nbins, phimax)
    philocmax_Vk1000_Rh5000, Pphilocmax_Vk1000_Rh5000 = (
        bins_Vk1000_Rh5000[np.argmax(hist_Vk1000_Rh5000)],
        np.max(hist_Vk1000_Rh5000))
    line_Vk1000_Rh5000 = plot.add_hist(
        bins_Vk1000_Rh5000, hist_Vk1000_Rh5000, peak=True)

    plot.ax.add_artist(plt.legend(handles=[
        Line2D([0], [0], color=line_Vj1000_Rg2000.get_color(),
            label=r'$(\tilde{v} = %.2e, \tilde{\nu}_r = %.2e,$' % (1e-2, 2e-5)
            + r'$\phi_{loc}^* = %1.2f, P(\phi_{loc}^* = %.2e)$'
            % (philocmax_Vj1000_Rg2000, Pphilocmax_Vj1000_Rg2000)),
        Line2D([0], [0], color=line_Vk1000_Rh5000.get_color(),
            label=r'$(\tilde{v} = %.2e, \tilde{\nu}_r = %.2e,$' % (1e-1, 5e-4)
            + r'$\phi_{loc}^* = %1.2f, P(\phi_{loc}^* = %.2e)$'
            % (philocmax_Vk1000_Rh5000, Pphilocmax_Vk1000_Rh5000))]))

    plot.fig.suptitle(r'$N = 1\cdot10^5, \phi = 0.80$' + '\n'
        + r'$S_{init}=%.2e, S_{max}=%.2e, N_{cases}=%.2e, l=%.2e$'
        % (5000, 1, 500, 10))

    plt.show()
