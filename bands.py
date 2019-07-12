#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jianzhou Zhao
@Date: 2019-05-11 21:40:31
@LastEditors: Jianzhou Zhao
@LastEditTime: 2019-07-12 14:31:21
@Description: File content
'''

import numpy as np
import mpiutils as mpi
from model import Model
from kpoint import Kpoint

if(mpi.rank == 0):
    import matplotlib.pyplot as plt

if mpi.rank == 0:
    tbmodel = Model.from_hr('graphene_ab.dat')
else:
    tbmodel = None

tbmodel = mpi.bcast(tbmodel, root=0)

hsp = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0/3.0, 1.0/3.0, 0.00000], [0.00000, 0.00000, 0.00000]], dtype=np.float)
kwant = 500
klist4bands = Kpoint.from_high_symmetry_points(khsym_frac=hsp, kwant=kwant)

kmpi = mpi.devide_array(klist4bands.kfrac, root=0)

empi = np.array([tbmodel.get_eigval(kvec) for kvec in kmpi])

print(mpi.rank, empi.shape)

eigs = np.zeros((klist4bands.nktot, tbmodel.nwan), dtype=np.float)
mpi.Gatherv(empi, (eigs, len(empi.flatten())), root=0)

print(mpi.rank, eigs.shape)

if(mpi.rank == 0):
    fig, ax = plt.subplots()
    for band in eigs.T:
        ax.plot(band)
    plt.savefig('bands.pdf')
