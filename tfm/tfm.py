#!/usr/bin/env python
"""
Compute Temporal Functional Modes
In other words, run a temporal ICA on the timeseries
of spatial ICA, and then remix the spatial maps to generate
'modes' of brain activity. For more information take a look at:

Smith 2012
Temporally-independent functional modes of spontaneous brain activity.
"""

import argparse
import ast
import logging
import os
import warnings

import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
from nilearn import image, plotting
import numpy as np
from sklearn.decomposition import FastICA

# warnings.filterwarnings("error")

ATLAS = '/project/3015046.07/atlas/Parcellations/MIST_444.nii.gz'


class MelodicData:

    def __init__(self, directory, labelfile='hand_classification.txt',
                 from_dr=False):

        self.from_dr = from_dr
        self.directory = directory
        self.mix = self.get_melodic_mix(directory, from_dr)
        self.n_components = self.mix.shape[1]

        if labelfile is not None:
            self.labels = self.get_labels(directory, labelfile)
        else:
            self.labels = list(range(self.n_components))

        self.signal = self.mix[:, self.labels]
        self.shape = self._get_rsns(from_dr).shape[:-1]
        self.affine = self._get_rsns(from_dr).affine

        self.explainedvar = self.get_explained_variance(directory)
        if not isinstance(self.explainedvar, int):
            self.explainedvar = self.explainedvar[self.labels]

    def get_melodic_mix(self, directory, from_dr=False):
        """Read in the spatial melodic mix.
        """
        fname = 'melodic_mix' if not from_dr else 'dr_stage1_subject00000.txt'
        mixfile = os.path.join(directory, fname)
        mix = np.loadtxt(mixfile)
        return mix

    def get_explained_variance(self, directory):
        """Read in the melodic_ICstats file
        """
        mixstats = os.path.join(directory, 'melodic_ICstats')
        if os.path.exists(mixstats):
            stats = np.loadtxt(mixstats)[:, 1]
            stats = stats[:, None]
        else:
            stats = 1
        return stats

    def get_labels(self, directory, labelfile):
        """Parse the classification file.
        """
        labelfile = os.path.join(directory, labelfile)
        with open(labelfile, 'r') as f:
            last_line = [line for line in f][-1]
        noise = [x - 1 for x in ast.literal_eval(last_line)]
        return [x for x in range(self.n_components)
                if x not in noise]

    def _get_rsns(self, from_dr=False):
        """Load melodic IC from directory.
        Note that we use the original ones, i.e.,
        melodic_oIC and not melodic_IC
        """
        if from_dr:
            fname = 'dr_stage2_subject00000.nii.gz'
        else:
            fname = 'melodic_oIC.nii.gz'
        return nib.load(os.path.join(self.directory,
                                     fname))

    @property
    def rsns(self):
        """
        Only returns signal components from melodic IC.
        """
        rsns = self._get_rsns(self.from_dr)
        temporal = rsns.shape[-1]
        data = np.reshape(rsns.get_data(), (-1, temporal))[:, self.labels]
        return data


class DenoisedData():

    def __init__(self, path, atlas=None):
        """
        Wrapper class for dataset so that it offers the same methods as
        the MelodicData class.
        """
        if atlas is None:
            atlas = ATLAS

        masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)
        signal = masker.fit_transform(path)

        atlasnifti = nib.load(atlas)
        atlasdata = nib.load(atlas).get_data()
        nrois = signal.shape[-1]
        rsns = np.stack([atlasdata.copy() for i in range(nrois)],
                        axis=-1)
        for index, volume in enumerate(np.rollaxis(rsns, -1)):
            # The idea of querying for index + 1 is because the
            # first map is the "non-relevant" non-brain area we want to skip.
            volume[atlasdata == index + 1] = 1
            volume[atlasdata != index + 1] = 0
        rsns = np.reshape(rsns, (-1, nrois))

        assert rsns.max() == 1
        assert rsns.min() == 0

        self.shape = atlasnifti.shape[:3]
        self.affine = atlasnifti.affine
        self.rsns = rsns
        self.signal = signal
        self.explainedvar = 1


class TFM:

    # Deflation takes much longer, but converges more often.
    def __init__(self, n_components=30, max_iter=20_000, tol=0.00001,
                 fun='logcosh', algorithm='parallel', random_state=None):

        self.ica = FastICA(max_iter=max_iter, tol=tol,
                           n_components=n_components,
                           fun=fun,
                           algorithm=algorithm,
                           random_state=random_state)

    def unmix(self, signal):
        """Call FastICA on the signal components.
        This is a separate function so we can call it from raicar.
        """
        sources = self.ica.fit_transform(signal)
        return sources

    def fit_transform_melodic(self, melodic_data):
        """Take a MelodicData object and unmix it.
        """
        sources = self.unmix(melodic_data.signal)
        rsns = melodic_data.rsns
        # Use a copy so we don't mutate the internals of FastICA
        mixing = self.ica.mixing_.copy()

        tfm = np.dot(rsns, mixing)

        # Mask outside of brain with NaN
        tfm[rsns.max(axis=-1) == 0] = np.nan

        # Demean and variance normalize *EACH COMPONENT INDIVIDUALLY.*
        tfm -= np.nanmean(tfm, axis=0)
        tfm /= np.nanstd(tfm, axis=0)

        tfm = np.nan_to_num(tfm)

        # Because ICA orientation is arbitrary, make it such that largest value
        # is always positive.
        for spatial_map, node_weights in zip(tfm.T, mixing.T):
            if spatial_map[np.abs(spatial_map).argmax()] < 0:
                spatial_map *= -1
                node_weights *= -1

        # Now order the components according to the RMS of the mixing matrix.
        weighted_mixing = melodic_data.explainedvar * mixing
        rms = np.sum(np.square(weighted_mixing), 0)
        order = np.argsort(rms)[::-1]  # Decreasing order.

        self.reordered_mixing = mixing[:, order]
        tfm = tfm[:, order]
        sources = sources[:, order]

        return nib.Nifti1Image(np.reshape(tfm,
                                          (*melodic_data.shape, -1)),
                               melodic_data.affine), sources


def main(args):

    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)

    if os.path.isabs(args.outputdir):
        outdir = args.outputdir
    else:
        outdir = os.path.join(os.path.dirname(args.inputdir), args.outputdir)

    def out(name):
        return os.path.join(outdir, name)

    # Check that input directories are OK and that we don't overwrite anything.
    if not args.dryrun:
        assert os.path.exists(args.inputdir)

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        else:
            if not args.force:
                assert os.listdir(outdir) == ""

    else:
        # Some debug info
        print(out('melodic_IC.nii.gz'))
        print(out('melodic_mix'))
        return

    # Load data
    logging.info(f"Loading Melodic Data from {args.inputdir}")

    # If inputdir is a directory:
    if os.path.isdir(args.inputdir):
        # Assume from dual_regression if directory name does not end in .ica
        from_dr = (not os.path.abspath(args.inputdir).endswith('.ica'))
        if args.no_label:
            melodic_data = MelodicData(args.inputdir, None, from_dr)
        else:
            melodic_data = MelodicData(args.inputdir, args.labelfile, from_dr)
    else:
        melodic_data = DenoisedData(args.inputdir)

    # Parse user inputs
    n_components = min(args.n_components, len(melodic_data.signal.T))
    logging.info(f"# of signal spatial ICs is {len(melodic_data.signal.T)}")
    tolerance = args.tolerance
    max_iter = args.max_iter

    # Compute TFMs.
    logging.info("Computing TFMs")
    try_counter = 1
    algorithm = 'parallel'
    while True:
        warnings.filterwarnings("error")
        try:
            # Try using the parallel algorithm because it's faster.
            tfm_ica = TFM(n_components=n_components,
                          max_iter=max_iter,
                          tol=tolerance,
                          algorithm=algorithm,
                          random_state=np.random.randint(0, 2**32 - 1))
            tfms, sources = tfm_ica.fit_transform_melodic(melodic_data)
        except UserWarning:
            try_counter += 1
            if try_counter > 5:
                logging.info("Parallel approach failed. Moving to Deflation")
                algorithm = 'deflation'
                try_counter = 1
            else:
                logging.info(f'ICA attempt {try_counter} using {algorithm}')
        else:
            logging.info(f"ICA successful after {try_counter} attempts.")
            break

    warnings.resetwarnings()

    # Save outputs
    logging.info(f"Saving outputs to directory {outdir}")
    tfms.to_filename(out('melodic_IC.nii.gz'))
    np.savetxt(out('melodic_unmix'), tfm_ica.reordered_mixing,
               delimiter='  ', fmt='%.6f')
    np.savetxt(out('melodic_mix'), sources, delimiter='  ', fmt='%.6f')
    np.savetxt(out('melodic_FTmix'), np.abs(np.fft.rfft(sources, axis=0)),
               delimiter='  ', fmt='%.6f')


def run_tfm():

    parser = _cli_parser()
    args = parser.parse_args()
    main(args)


def _cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    # output directory
    parser.add_argument('inputdir', type=str,
                        help='Input directory with melodic data, or input file for atlas-based tICA.')

    parser.add_argument('-o', '--outputdir', type=str, default='tfm.ica',
                        help='Directory where results will be stored. Def: tfm.ica')

    parser.add_argument('-l', '--labelfile', type=str,
                        default='hand_classification.txt',
                        help='Name of classification file. Default hand_classification.txt')

    parser.add_argument('--n_components', type=int, default=None,
                        help='Number of components to extract from tICA decomposition')

    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='ICA tolerance. Default 1e-4')

    parser.add_argument('--max_iter', type=int, default=10_000,
                        help='ICA max number of iterations')

    parser.add_argument('--dryrun', action='store_true',
                        help='Print which files would be generated.')

    parser.add_argument('--force', action='store_true',
                        help='Overwrite files.')

    parser.add_argument('--no-label', action='store_true',
                        help='Consider all components to be signals.')


    return parser


if __name__ == '__main__':
    run_tfm()
