# -*- coding: utf-8 -*-
# @Author: jzzhao
# @Email:  jianzhou.zhao@gmail.com
# @Date:   2017-11-19 16:33:39
# @Last Modified by:   jzzhao
# @Last Modified time: 2017-12-21 19:01:37

import numpy as np

class rpoint:

    def __init__(self, name='real points'):
        self.name = name

    # read real space points from wannier_hr file
    def from_file(self,file):

        from pandas import read_table

        with open(file, 'r') as wfile:

            self.name  = wfile.readline()
            self.nwan  = int( wfile.readline() )
            self.nrpt  = int( wfile.readline() )

            if (self.nrpt%15) :
                nline = self.nrpt//15 + 1
            else :
                nline = self.nrpt//15

            # get point weight
            rwt = []
            [ [rwt.append(float(x)) for x in next(wfile).split()] for x in range(nline) ]
            self.weight = np.array(rwt); del rwt

        # read hamiltonian
        rdat = np.array( read_table(file, skiprows=3+nline, delim_whitespace=True, header=None) )
        self.ham  = rdat[:, 5].reshape(self.nrpt,self.nwan,self.nwan).transpose((0,2,1)) \
          + 1j*rdat[:, 6].reshape(self.nrpt,self.nwan,self.nwan).transpose((0,2,1))

        # get rpt position
        self.pos = rdat[::self.nwan*self.nwan, :3]

        # get onsite logical variable
        self.onsite = np.full(self.nrpt,False)

        # get onsite index
        self.onsite[np.where( np.all( self.pos == [0.0,0.0,0.0], axis=1) )[0][0]] = True

        return self

    def change_basis(self, initial='r', target='r', index=[[]], orb=None, add_spin=False):

        # map the path to int number
        path_switch = {
            'r': 0,
            'c': 1,
            'j': 2
        }
        # map the orbital to ll
        orb_switch = {
            's': 0,
            'p': 1,
            'd': 2,
            'f': 3,
            't':-1
        }

        asize = []
        for inum, item in enumerate(index):
            atmp = item[1]-item[0]
            norb_from_orb = 2*abs( orb_switch[orb[inum][0]] )+1
            if atmp == norb_from_orb :
                spin_orig = False
            elif atmp == 2*norb_from_orb :
                spin_orig = True
            else :
                exit('Wrong options in change_basis')
            print(atmp, norb_from_orb, spin_orig)

        op = -1 * (path_switch[target] - path_switch[initial]) // abs(path_switch[target] - path_switch[initial])
        print(op)
        path = list(range(path_switch[target], path_switch[initial], op))
        print(path)


