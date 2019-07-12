# @Author: Jianzhou Zhao <jzzhao>
# @Date:   2018-11-21T12:05:50+01:00
# @Email:  jianzhou.zhao@gmail.com
# @Last modified by:   jzzhao
# @Last modified time: 2018-11-21T16:37:10+01:00

import numpy as np
import scipy.linalg as la
from collections import defaultdict


class Model():

    def __init__(self, nwan=None, nrpt=None, deg=None, ham=None):
        self.nwan = nwan
        self.nrpt = nrpt
        self.deg = deg
        self.ham = ham
        return

    @classmethod
    def from_hr(cls, hr_file):

        with open(hr_file, 'r') as f:
            next(f)  # skip title line
            nwan = int(next(f))
            nrpt = int(next(f))

            # read degeneration of R points
            deg = []
            for _, line in zip(range(int(np.ceil(nrpt / 15))), f):
                deg.extend(int(i) for i in line.split())

            assert len(deg) == nrpt

            # read rpt and hamr
            raw_list = [line.split() for line in f]

            nwan_square = nwan**2
            hr_list = defaultdict(list)
            for num, line in enumerate(raw_list):
                Rvec = tuple(map(int, line[:3]))
                hr_list[Rvec].append(
                    (float(line[5]) + 1j * float(line[6])) / deg[num // nwan_square])

        for key, array in hr_list.items():
            hr_list[key] = [array[iwan::nwan] for iwan in range(nwan)]

        return cls(nwan=nwan, nrpt=nrpt, deg=deg, ham=hr_list)

    def get_bulk_Hk(self, Kvec):
        return np.sum([np.array(Hvec, dtype=np.complex) * np.exp(2j * np.pi * np.dot(Rvec, Kvec))
                       for Rvec, Hvec in self.ham.items()], axis=0)

    def get_eigval(self, Kvec):
        return la.eigvalsh(self.get_bulk_Hk(Kvec))


if __name__ == '__main__':

    c6 = Model().from_hr('wannier90_hr.dat')

    print(c6.get_eigval([0, 0, 0]))
