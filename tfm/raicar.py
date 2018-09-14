"""The original RAICAR code worked with a custom made fastica
code with the following signature:

A, W, S = fastica(X, nSources), where
X - size n x p, n samples, p mixtures
nSources - number of components to return.

it outputs:

A mixing matrix (n samples, nSources)
W unmixing matrix (nSources, n samples)
S sources (nSources, p mixtures)

The signature is the same as scikit learn's. The main difference
is that scikit-learn does not return A, W and S the same way.

"""
import os
import glob
import numpy as np
from pycar.raicar import RAICAR

from tfm import TFM, MelodicData


def ica_method(x, nSources=10, **kwargs):
    tfm_ica = TFM(nSources, tol=1e-5, fun='logcosh')
    sources = tfm_ica.unmix(x)
    mixing = tfm_ica.ica.mixing_
    unmixing = tfm_ica.ica.components_
    return mixing, unmixing, sources.T


def main():

    raicar = RAICAR(projDirectory='raicar', nSignals=10, icaMethod=ica_method)
    raicar.clean_project()
    x = np.loadtxt('melodic_mix').T
    raicar.runall(x)

directories = glob.glob('/project/3015046.07/derivatives/resting-state/fmriprep/sub-*/ses-*/func/sub-*_ses-*_task-rest_run-*_bold_space-MNI152NLin2009cAsym_preproc_highpass100.ica')

dirgroups = glob.glob('/project/3015046.07/derivatives/resting-state/subject-concat-ica/sub-*_group.ica')


def run(index, ncomponents):
    for directory in dirgroups[index:index+1]:
        os.chdir(directory)
        raicar = RAICAR(projDirectory='raicar',
                        nSignals=ncomponents, icaMethod=ica_method)
        melodic_data = MelodicData(directory)
        # raicar.clean_project()
        raicar.runall(melodic_data.signal)
        return raicar
