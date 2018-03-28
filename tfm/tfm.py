#!/usr/bin/env python
"""
Compute Temporal Functional Modes
"""

import ast
import os

import nibabel as nib
import numpy as np
from sklearn.decomposition import FastICA


class TFM:

    def __init__(self, directory, n_components=30,
                 max_iter=10_000, tol=0.0001,
                 labelfile='hand_classification.txt'):

        self.directory = directory
        self.mix = self.get_melodic_mix(directory)
        self.n_sica_components = self.mix.shape[1]
        self.signals = self.get_signals(directory, labelfile,
                                        self.n_sica_components)
        self.ica = FastICA(max_iter=max_iter,
                           tol=tol,
                           n_components=n_components)

    def get_melodic_mix(self, directory):
        mixfile = os.path.join(directory, 'melodic_mix')
        mix = np.loadtxt(mixfile)
        return mix

    def get_signals(self, directory, labelfile):
        labelfile = os.path.join(directory, labelfile)
        with open(labelfile, 'r') as f:
            last_line = [line for line in f][-1]
        noise = [x - 1 for x in ast.literal_eval(last_line)]
        return [x for x in range(self.n_components)
                if x not in noise]

    def _get_rsns(self):
        return nib.load(os.path.join(self.directory,
                                     'melodic_IC.nii.gz'))

    def _unmix(self):
        sources = self.ica.fit_transform(self.mix[:, self.labels])
        return sources, self.ica.mixing_

    def generate_tfms(self):
        sources, mixing_matrix = self._unmix()
        rsns = self._get_spatial_icas()
        spatial = rsns.shape[:-1]
        temporal = rsns.shape[-1]
        data = np.reshape(rsns.get_data(), (-1, temporal))[:, self.signals]
        tfm = np.dot(data, mixing_matrix)
        return nib.Nifti1Image(np.reshape(tfm,
                                          (*spatial, -1)),
                               rsns.affine), sources


def main():

def run_tfm():

    parser =_cli_parser()
    args = parser.parse_args()
    main(args)


def _cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)


if __name__ == '__main__':
    run_tfm()

