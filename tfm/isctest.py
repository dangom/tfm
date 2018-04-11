"""
Wrapper to Matlab's ISCTEST
ISCTEST: Testing independent components by intersession/subject consistency

Version 2.2, Feb 2013
Aapo Hyvarinen, University of Helsinki

clustering=ISCTEST(spatialPattTens,alphaFP,alphaFD,'components') tests
  the matrices of independent components (e.g. spatial patterns in fMRI),
  previously estimated by ICA, for consistency
  over the different data sets (subjects/sessions) from which they were
  estimated. It returns the set of clusters which are considerent
  statistically significant.

clustering=ISCTEST(spatialPattTens,alphaFP,alphaFD,'mixing') tests
  the columns of the mixing matrix instead of the independent components.

See http://www.cs.helsinki.fi/u/ahyvarin/code/isctest/ for more information
"""
import re
import os

import matlab
import matlab.engine
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def isctest(spatial_patterns, *, fpr=0.005, fdr=0.05,
            test='components', engine=None):
    """
    spatial_patterns: 3-D tensor which gives the vectors to be tested. Each
     matrix spatialPattTens(:,:,k) is the matrix of the spatial patterns for
     the k-th subject. In the case of ICA, it is usually the ICA results in the
     following sense:
    1. For temporal ICA (MEG/EEG), spatialPattTens(:,:,k) is the mixing matrix A
    2. For spatial ICA (fMRI), spatialPattTens(:,:,k) is the matrix S transposed
    Thus, the indices are: channel(E/MEG)/voxel(fMRI), component, subject
    """
    if engine is None:
        raise RuntimeError('Missing matlab engine')

    if test not in ('mixing', 'components'):
        raise NameError('Only \'mixing\' and \'components\' are acceptable tests.')

    data = matlab.double(spatial_patterns.tolist())
    res = engine.isctest(data, fpr, fdr, test)
    # Convert from matlab back to python before returning
    return np.array(list(map(list, list(res)))).astype(int) - 1


def visualize_cluster(datafiles, iscresults, cluster=0, ax=None):
    data = [np.loadtxt(file)[:, comp]
            for file, comp in zip(datafiles, iscresults[cluster])
            if comp >= 0]
    if ax is None:
        fig, ax = plt.subplots()

    ax.pcolor(np.stack(data))


def visualize_all_clusters(data, res):
    fig, axes = plt.subplots(nrows=3, ncols=3)
    for index, ax in enumerate(axes.flatten()):
        visualize_cluster(data, res, index, ax=ax)


def save_cluster_components(filename):
    def patch_filename(inname):
        return os.path.join(os.path.dirname(inname),
                            'melodic_IC.nii.gz')

    arrs = [(nib.load(patch_filename(f)), c)
            for f, c in zip(data, res[0]) if c >= 0]
    affine = arrs[0][0].affine
    header = arrs[0][0].header
    out = [x.get_data()[:, :, :, y] for x, y in arrs]
    niftiout = nib.Nifti1Image(np.stack(out, axis=-1),
                               affine, header)
    nib.save(niftiout, filename)
