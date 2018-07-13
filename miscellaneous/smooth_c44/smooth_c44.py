"""
This script shows the relevance of the gaussian smoothing of the projection of
cos(4 \\theta) of strain correlations (C44) in a special case. (see
https://yketa.github.io/UBC_2018_Wiki/#Smoothing%20strain%20correlations)

File c44_to_smooth.pickle contains a list of radii divided by average particle
separation and the corresponding list of values of C44 at these radii.

Execution of this script necessitates the active_particles package (see
https://github.com/yketa/active_particles).
"""

import pickle

from numpy import sqrt, arange

from active_particles.maths import gaussian_smooth_1D

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if __name__ == '__main__':

    # VARIABLES

    N = 1e5             # number of particles used
    Ncases = 300        # number of square boxes in one dimensions used for displacement grid computation
    dL = sqrt(N)/Ncases # length of each square box in units of average particle separation

    with open('c44_to_smooth.pickle', 'rb') as c44_file:
        x_c44, c44 = pickle.load(c44_file)

    # PLOT

    mpl.rcParams.update({'font.size': 15})

    fig, ax = plt.subplots()

    fig.suptitle(
        r'$N=10^5, \phi=0.80, \tilde{v}=10^{-2}, \tilde{\nu}_r=2\cdot10^{—5}$'
        + r'$, \Delta t=10^3, nD_0\Delta t=6.28\cdot10^2$' + '\n'
        + r'$S_{init}=5\cdot10^3, S_{max}=5\cdot10^2, N_{cases}=3\cdot10^2$'
        + r'$, dL/a=%.2e$' % dL)
    ax.set_xlabel(r'$r/a$')
    ax.set_ylabel(r'$C_4^4(r) = \frac{1}{\pi}\int_0^{2\pi}d\theta$'
		+ ' ' + r'$C_{\varepsilon_{xy}\varepsilon_{xy}}(r, \theta)$'
		+ ' ' + r'$\cos4\theta$')

    ax.set_xscale('log')
    ax.set_yscale('log')

    lines = []
    for sigma in (0, 0.1, 0.2, 0.5, 1, dL): # smoothing length scale
        lines += ax.plot(x_c44, gaussian_smooth_1D(x_c44, c44, sigma),
            label=r'$\sigma_{smooth}/a=%1.2f$' % sigma)

    dL_multiples = arange(max(x_c44), step=dL)
    for multiple in dL_multiples:   # multiples of dL
        if multiple >= min(x_c44):
            ax.axvline(multiple, color='black', linestyle='--', linewidth=0.5)
    lines.append(Line2D([0], [0], color='black', linestyle='--', linewidth=0.5,
        label='multiples of ' + r'$dL$'))

    legend = plt.legend(handles=lines)
    ax.add_artist(legend)

    plt.show()
