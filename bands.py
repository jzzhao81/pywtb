#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jianzhou Zhao
@Date: 2019-05-11 21:40:31
@LastEditors: Jianzhou Zhao
@LastEditTime: 2019-07-12 17:09:16
@Description: File content
'''


def plotbands(kpth, eigs, file='bands.pdf', dpi=100, ylim=None, ticks=None, labels=None, linewidth=1.0,
              color='red', figsize=(6, 6), ef=0.0):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for band in eigs:
        ax.plot(kpth, band, linewidth=linewidth, color=color)
    ax.set_xlim(kpth[0], kpth[-1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if ticks is not None:
        ax.set_xticks(ticks)
    if ticks is not None and labels is not None:
        ax.set_xticklabels(labels)
    ax.axhline(ef, linestyle='--', color='grey')
    for tick in ticks[1:-1]:
        ax.axvline(tick, linestyle='--', color='grey')
    plt.savefig(file, dpi=dpi)
    return ax


if __name__ == "__main__":

    from kpoint import Kpoint
    from model import Model
    import mpiutils as mpi
    import numpy as np

    tbmodel = Model.from_hr('graphene_ab.dat')

    hskp = [[0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0/3.0, 1.0/3.0, 0.00000],
            [0.00000, 0.00000, 0.00000]]
    kwant = 500
    kpts = Kpoint.from_high_symmetry_points(khsym_frac=hskp, kwant=kwant)
    eigs = tbmodel.get_eigvals(kpts.kfrac)
    ticks = Kpoint.k_length_accumulate((np.array(hskp)))
    label = ['$\Gamma$', 'X', 'K', '$\Gamma$']

    if mpi.rank0:
        plotbands(Kpoint.k_length_accumulate(kpts.kreal), eigs, ticks=ticks, labels=label)
