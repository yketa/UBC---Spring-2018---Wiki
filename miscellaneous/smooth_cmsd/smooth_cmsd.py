"""
This script shows the relevance of the Gaussian smoothing of the shear strain
correlations via the collective mean square displacements in a special case.
(see
https://yketa.github.io/UBC_2018_Wiki/#Gaussian%20coarse-graining%20and%20CMSD)

File param.p contains simulation parameters.

File Cttb_Dk8000_Vj1000_Rg2000_Nq1000_Io5000_Tl1000_Mn5000_Cn5000.pickle
contains grid of wave vectors, grid of mean square norms of cross products of
normalised wave vectors with displacement grids Fourier transform and
cylindrical average of the latter.

File Cllb_Dk8000_Vj1000_Rg2000_Nq1000_Io5000_Tl1000_Mn5000_Cn5000.pickle
contains grid of wave vectors, grid of mean square norms of dot products of
normalised wave vectors with displacement grids Fourier transform and
cylindrical average of the latter.

Execution of this script requires the active_particles package (see
https://github.com/yketa/active_particles).
"""

import os
os.environ['SHOW'] = 'True'

import pickle

import numpy as np

from active_particles.quantities import nD0_active

from active_particles.analysis.ctt import StrainCorrelationsCMSD
from active_particles.plot.plot import list_colormap
from active_particles.plot.mpl_tools import FittingLine

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

if __name__ == '__main__':

    script_path = os.path.dirname(os.path.realpath(__file__))   # path to this script

    # VARIABLES

    init_frame=5000 # frame considered as initial
    dt = 1          # lag time in number of frames
    int_max = 500   # number of frames considered in the calculation
    Ncases = 500    # number of boxes in each direction of the displacement grid

    with open(os.path.join(script_path, 'param.p'), 'rb') as param_file:
        parameters = pickle.load(param_file)                    # simulation parameters
    av_p_sep = parameters['box_size']/np.sqrt(parameters['N'])  # average particle separation
    nD0 = nD0_active(parameters['N'], parameters['vzero'], parameters['dr'],
        parameters['box_size'])                                 # product of particle density and active diffusion constant

    r_cut_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1]   # list of wave length Gaussian cut-off radii in units of average particle separation

    with open(os.path.join(script_path,
        'Cttb_Dk8000_Vj1000_Rg2000_Nq1000_Io5000_Tl1000_Mn5000_Cn5000.pickle'),
        'rb') as Ctt_file:
        wave_vectors, k_cross_FFTugrid2D_sqnorm, _ = pickle.load(Ctt_file)  # grids of wave vectors and mean square norms of cross products of normalised wave vectors with displacement grids Fourier transform
    with open(os.path.join(script_path,
        'Cllb_Dk8000_Vj1000_Rg2000_Nq1000_Io5000_Tl1000_Mn5000_Cn5000.pickle'),
        'rb') as Cll_file:
        _, k_dot_FFTugrid2D_sqnorm, _ = pickle.load(Cll_file)               # grid of mean square norms of dot products of normalised wave vectors with displacement grids Fourier transform

    # CALCULATION

    sc = StrainCorrelationsCMSD(wave_vectors,
        k_cross_FFTugrid2D_sqnorm, k_dot_FFTugrid2D_sqnorm) # strain correlations object

    Css = {}                                # hash table of shear strain correlations with Gaussian cut-off radii as keys
    filtered_k_cross_FFTugrid1D_sqnorm = {} # hash table of filtered mean square norms of cross products of normalised wave vectors with displacement grids Fourier transform
    filtered_k_dot_FFTugrid1D_sqnorm = {}   # hash table of filtered mean square norms of dot products of normalised wave vectors with displacement grids Fourier transform

    r_cut_list.sort()
    for r_cut in r_cut_list:
        Css[r_cut] = sc.strain_correlations(r_cut=r_cut*av_p_sep)
        filtered_k_cross_FFTugrid1D_sqnorm[r_cut] = (sc
            .filtered_k_cross_FFTugrid1D_sqnorm)
        filtered_k_dot_FFTugrid1D_sqnorm[r_cut] = (sc
            .filtered_k_dot_FFTugrid1D_sqnorm)

    # PLOT PARAMETERS

    mpl.rcParams.update({'font.size': 13})

    colors = list_colormap(r_cut_list)  # line colors hash table

    y_min = 1e-4    # minimum y-coordinate value
    y_max = 1e7     # maximum y-coordinate value

    x_label = r'$\lambda/a = 2\pi/|\vec{k}|a$'  # x-axis label

    # PLOT

    fig = plt.figure()
    fig.suptitle(
        r'$N=%.2e, \phi=%1.2f, \tilde{v}=%.2e, \tilde{\nu}_r=%.2e,$'
		% (parameters['N'], parameters['density'], parameters['vzero'],
		parameters['dr'])
		+ r'$\Delta t=%.2e, nD_0 \Delta t=%.2e$'
		% (dt*parameters['period_dump']*parameters['time_step'],
		nD0*dt*parameters['period_dump']*parameters['time_step'])
		+ '\n' +  r'$S_{init}=%.2e$' % init_frame
		+ r'$, S_{max}=%.2e, N_{cases}=%.2e$' % (int_max, Ncases)
        + '\n' + r'$\tilde{\mathcal{G}}^2_{r_{cut}}(\vec{k}) \equiv$'
		+ r'$\exp(-r_{cut}^2 |\vec{k}|^2)$')
    fig.subplots_adjust(wspace=0.25)

    gs = GridSpec(1, 3, width_ratios=[1, 1, 2/10])

    ax_cross = plt.subplot(gs[0])
    ax_cross.set_ylim([y_min, y_max])
    ax_cross.set_ylabel(
        r'$\left<||\vec{k}\wedge\tilde{\vec{u}}(\vec{k})||^2\right>/k^2$'
        + r'$\times \tilde{\mathcal{G}}^2_{r_{cut}}(\vec{k})$')
    ax_cross.set_xlabel(x_label)

    ax_dot = plt.subplot(gs[1])
    ax_dot.set_ylim([y_min, y_max])
    ax_dot.set_ylabel(
        r'$\left<||\vec{k}\cdot\tilde{\vec{u}}(\vec{k})||^2\right>/k^2$'
        + r'$\times \tilde{\mathcal{G}}^2_{r_{cut}}(\vec{k})$')
    ax_dot.set_xlabel(x_label)

    for r_cut in r_cut_list:
        ax_cross.loglog(
            2*np.pi/filtered_k_cross_FFTugrid1D_sqnorm[r_cut][1:, 0]/av_p_sep,
            filtered_k_cross_FFTugrid1D_sqnorm[r_cut][1:, 1],
            color=colors[r_cut])
        ax_dot.loglog(
            2*np.pi/filtered_k_cross_FFTugrid1D_sqnorm[r_cut][1:, 0]/av_p_sep,
            filtered_k_dot_FFTugrid1D_sqnorm[r_cut][1:, 1],
            color=colors[r_cut])

    ax_legend = plt.subplot(gs[2])
    ax_legend.axis('off')
    ax_legend.legend(handles=list(map(
        lambda r_cut: Line2D([0], [0], color=colors[r_cut],
            label=r'$r_{cut}/a=%.2e$' % r_cut),
        r_cut_list)),
        loc='center')

    fl_cross = FittingLine(ax_cross, 2, 0, 4, y_fit='', x_fit='(\\lambda/a)')   # cross product axes fitting line
    fl_dot = FittingLine(ax_dot, 2, 0, 4, y_fit='', x_fit='(\\lambda/a)')       # dot product axes fitting line

    plt.show()
