"""
This script plots local packing fraction distributions and mean square
displacements at (phi = 0.80, vzero = 1e-2, dr = 2e-5) and
(phi = 0.80, vzero = 1e-1, dr = 5e-4). (see
https://yketa.github.io/UBC_2018_Wiki/#Strain%20correlations%20at%20high%20activity)

File varN_Vj1000_Rg2000.pickle contains a list of densities at (phi = 0.80,
vzero = 1e-2, dr = 2e-5) and file varN_Vk1000_Rh5000.pickle contains a list of
densities at (phi = 0.80, vzero = 1e-1, dr = 5e-4).

Files msd_Vj1000_Rg2000_I[initial frame index].csv contain mean square
displacements at (phi = 0.80, vzero = 1e-2, dr = 2e-5) computed with different
initial frames. Files msd_Vk1000_Rh5000_I[initial frame index].csv contain mean
square displacements at (phi = 0.80, vzero = 1e-1, dr = 5e-4) computed with
different initial frames.

Execution of this script requires the active_particles package (see
https://github.com/yketa/active_particles).
"""

import os

import pickle

import numpy as np

from active_particles.init import set_env
set_env('SHOW', True)
from active_particles.exponents import float_to_letters

from active_particles.analysis.varn import histogram, Plot
from active_particles.plot.plot import list_linestyles

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if __name__ == '__main__':

    # VARIABLES

    Nbins = 100     # number of bins for the histogram
    phimax = 1.4    # maximum local density for histogram

    init = [500, 1000, 2000, 5000]  # initial frame indexes in mean square displacement computations

    dr = {'Vj1000_Rg2000': 2e-5, 'Vk1000_Rh5000': 5e-4} # rotation diffusion rates corresponding to abbreviated directory names

    # PLOT PARAMETERS

    mpl.rcParams.update({'font.size': 14})

    # LOCAL PACKING FRACTION PLOT

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
            label=r'$\tilde{v} = %.2e, \tilde{\nu}_r = %.2e,$' % (1e-2, 2e-5)
            + r'$\phi_{loc}^* = %.2f, P(\phi_{loc}^*) = %.2e,$'
            % (philocmax_Vj1000_Rg2000, Pphilocmax_Vj1000_Rg2000)
            + r'$\int d\phi_{loc}$' + ' ' + r'$P(\phi_{loc})\phi_{loc} = %.3f$'
            % (np.mean(densities_Vj1000_Rg2000))),
        Line2D([0], [0], color=line_Vk1000_Rh5000.get_color(),
            label=r'$\tilde{v} = %.2e, \tilde{\nu}_r = %.2e,$' % (1e-1, 5e-4)
            + r'$\phi_{loc}^* = %.2f, P(\phi_{loc}^*) = %.2e,$'
            % (philocmax_Vk1000_Rh5000, Pphilocmax_Vk1000_Rh5000)
            + r'$\int d\phi_{loc}$' + ' ' + r'$P(\phi_{loc})\phi_{loc} = %.3f$'
            % (np.mean(densities_Vk1000_Rh5000)))]))

    plot.fig.suptitle(r'$N = 1\cdot10^5, \phi = 0.80$' + '\n'
        + r'$S_{init}=%d, S_{max}=%d, N_{cases}=%d, l=%d$'
        % (5000, 10, 500, 10))

    # MEAN SQUARE DISPLACEMENT PLOT

    fig, ax = plt.subplots()

    ax.set_xlabel(r'$\tilde{\nu}_r \Delta t$')
    ax.set_ylabel(r'$\left<|\Delta\vec{r}(\Delta t)|^2\right>/\Delta t$')

    msd = {'Vj1000_Rg2000':{}, 'Vk1000_Rh5000': {}}
    colors = {'Vj1000_Rg2000': line_Vj1000_Rg2000.get_color(),
        'Vk1000_Rh5000': line_Vk1000_Rh5000.get_color()}
    linestyles = list_linestyles(init)

    for dir in msd:
        for ini in init:
            msd[dir][ini] = np.genfromtxt(
                'msd_%s_I%s.csv' % (dir, float_to_letters(ini)),
                delimiter=',', skip_header=True)
            ax.loglog(
                dr[dir]*msd[dir][ini][:, 0],
                msd[dir][ini][:, 1]/msd[dir][ini][:, 0],
                color=colors[dir], linestyle=linestyles[ini])

    ax.add_artist(plt.legend(ncol=2, handles=
        [Line2D([0], [0], color=colors['Vj1000_Rg2000'],
            label=r'$\tilde{v} = %.2e, \tilde{\nu}_r = %.2e$' % (1e-2, 2e-5)),
        Line2D([0], [0], color=colors['Vk1000_Rh5000'],
            label=r'$\tilde{v} = %.2e, \tilde{\nu}_r = %.2e$' % (1e-1, 5e-4))]
        + list(map(
            lambda ini: Line2D([0], [0],
                color='black', linestyle=linestyles[ini],
                label=r'$S_{init} = %d$' % ini),
            init))))

    fig.suptitle(r'$N = 1\cdot10^5, \phi = 0.80$' + '\n'
        + r'$S_{max} = %d, S_{period} = %d$' % (1, 50))

    plt.show()
