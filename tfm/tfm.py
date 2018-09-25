"""
Compute Temporal Functional Modes
In other words, run a temporal ICA on the timeseries
of spatial ICA, and then remix the spatial maps to generate
'modes' of brain activity. For more information take a look at:

Smith 2012
Temporally-independent functional modes of spontaneous brain activity.

TODO:
1. Save logs to a file. Document inputs to command and TFM version.
2. Save a report with images from TFMs, their timecourses and (possibly)
labels.
"""

import argparse
import ast
import logging
import os
import os.path as op
import sys
import warnings

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import FastICA

from nilearn.input_data import NiftiLabelsMasker

from . import __version__

MIST_ROOT = op.join(op.dirname(__file__), 'mistatlas')
MIST_ATLAS_444 = op.join(MIST_ROOT, 'Parcellations/MIST_444.nii.gz')
MIST_HIERARCHY = op.join(MIST_ROOT, 'Hierarchy/MIST_PARCEL_ORDER.csv')


def mist_parcel_info(nrois):
    return op.join(MIST_ROOT, f'Parcel_Information/MIST_{nrois}.csv')


# TODO: Perhaps return a dataframe with original roi, target roi and labels makes more sense.
def atlas_parcel_labels(nrois, target_res=12):
    """This function takes two integer numbers, nrois and target_res. nrois
    indicates the number of MIST parcels used for an atlas based tICA
    decomposition, and target_res indicates the number of MIST parcels (higher
    in the MIST hierarchy) that label each of the original nrois.

    The idea is to group ROIs into higher level networks only for labelling and
    summarizing.
    """
    hierarchy = pd.read_csv(MIST_HIERARCHY, delimiter=',')
    roi_ids = np.unique(hierarchy[f's{nrois}'].values)
    rois_parent = [np.unique(hierarchy[f's{target_res}'][hierarchy[f's{nrois}'] == x]) for x in roi_ids]
    # Because all values are the same, take the first one. Use tolist() to get
    # a single flat list in the end.
    rois_parent = [x.tolist()[0] for x in rois_parent]
    labels = pd.read_csv(mist_parcel_info(12), delimiter=';')
    parent_names = [labels[labels['roi'] == x]['name'].values.tolist()[0] for x in rois_parent]
    parent_labels = [labels[labels['roi'] == x]['label'].values.tolist()[0] for x in rois_parent]
    return rois_parent, parent_names, parent_labels


def labeled_unmix(unmix):
    """unmix is the filename of a melodic_unmix matrix, i.e., the "core" tICA
    mixing matrix. This routine takes the filename and returns a pandas
    DataFrame with columns: label, name, roi (numbered parcels), tfm (numbered
    tfms), coefficient (roi x tfm entry), abs_coefficient
    (np.abs(coefficient)). """
    unmix_df = pd.DataFrame(np.loadtxt(unmix))
    rois_parent, parent_names, parent_labels = atlas_parcel_labels(len(unmix_df))
    unmix_df['name'] = parent_names
    unmix_df['label'] = parent_labels
    unmix_df = unmix_df.reset_index().melt(['index', 'label', 'name'])
    unmix_df = unmix_df.rename(columns={'index': 'roi',
                                        'variable': 'tfm',
                                        'value': 'coefficient'})
    unmix_df['abs_coefficient'] = unmix_df['coefficient'].abs()
    return unmix_df


def heatmap(filename, ax):
    """Plot the core TFM mixing matrix as a heatmap, which ROIs contributions
    aggregated by the 12 MIST functional networks.
    """
    data_raw = np.abs(pd.DataFrame(np.loadtxt(filename)))
    _, parent_names, _ = atlas_parcel_labels(len(data_raw))
    data_raw['name'] = parent_names
    data = data_raw.groupby('name').aggregate(sum)

    # Normalise TFMs such that contributions of all networks sum to 100%
    for column in data:
        data[column] = 100*data[column]/np.sum(data[column])

    g = sns.heatmap(data, yticklabels=1,
                    annot=True, fmt=".1f", linewidth=.5, ax=ax,
                    vmin=0, vmax=25)

    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    return g


def parse_melodic_labelfile(labelfile):
    """Utility method to parse the IC classification file, as per
    the conventions of FIX and FSLEYES.
    """
    with open(labelfile, 'r') as f:
        lines = [line for line in f]
        penultimate_line, last_line = lines[-2:]

    n_components = int(penultimate_line.split(',')[0])

    # In the labelfile, the first component is numbered 1
    noise = [x - 1 for x in ast.literal_eval(last_line)]
    return [x for x in range(n_components) if x not in noise]


def atlas_roitovol(atlas, nrois):
    """Some atlases are 3D files wherein each value represents
    a separate ROI. For example, 0 for non-brain, 1 for CSF, 2 for
    gray matter, etc.
    This function takes such a 3D atlas an returns it in 4D format,
    where each volume contains a single ROI encoded as a binary mask.

    TODO: The number of ROIS can easily be inferred from the data
    itself, but int(max(atlas.get_data())) fails for a small subset
    of atlas tested.
    """
    atlasnifti = nib.load(atlas)
    atlasdata = atlasnifti.get_data()
    maps = np.stack([atlasdata.copy() for i in range(nrois)],
                    axis=-1)

    # Loop through the last (4th) dimension of maps.
    for index, volume in enumerate(np.rollaxis(maps, -1)):
        # The idea of querying for index + 1 is because the
        # first map is the "non-relevant" non-brain area we want to skip.
        offset = 1
        volume[atlasdata == index + offset] = 1
        volume[atlasdata != index + offset] = 0

    assert (maps.max() == 1 and maps.min() == 0)

    return nib.Nifti1Image(maps, atlasnifti.affine)


class Data:
    """In the TFM / temporal ICA model, the input data to be decomposed
    consists of a matrix of dimensions n_timepoints x n_mixtures. Mixtures is
    an abstract concept here, but when talking about fMRI we think of mixtures
    as voxels, or ROIs or even resting state networks. In addition to the input
    data matrix, the TFM model also requires a set of spatial maps as inputs,
    where each spatial map is associated to one of the n_mixtures mentioned
    about. The spatial maps are thus matrices of dimensions n_voxels x
    n_mixtures.
    """
    def __init__(self, timeseries, maps, kind=None):
        """Timeseries is a numpy array.
        Maps is a nibabel Nifti1Image object.
        """
        assert timeseries.shape[-1] == maps.shape[-1]
        self.timeseries = timeseries
        self.maps = maps

        # here for compatibility with previous versions
        self.shape = maps.shape[:-1]
        self.affine = maps.affine
        self.explainedvar = 1

        self.kind = None

    @property
    def rsns(self):
        temporal = self.maps.shape[-1]
        data = np.reshape(self.maps.get_data(), (-1, temporal))
        return data

    @property
    def signal(self):
        return self.timeseries

    @classmethod
    def from_melodic(cls, icadir, labelfile=None):
        """Load data from an FSL melodic directory.
        Assume labelfile is relative to the ica directory.
        """
        melodic_mix = np.loadtxt(op.join(icadir, 'melodic_mix'))
        try:
            melodic_oic = nib.load(op.join(icadir, 'melodic_oIC.nii.gz'))
        except FileNotFoundError:
            # When running MELODIC from the GUI, melodic oIC may not exist.
            logging.warning('Melodic oIC not found. Using IC.')
            melodic_oic = nib.load(op.join(icadir, 'melodic_IC.nii.gz'))

        if labelfile is not None:
            labels = parse_melodic_labelfile(op.join(icadir, labelfile))
            melodic_mix = melodic_mix[:, labels]
            mapdata = melodic_oic.get_data()
            melodic_oic = nib.Nifti1Image(mapdata[:, :, :, labels],
                                          melodic_oic.affine)

        return cls(timeseries=melodic_mix, maps=melodic_oic, kind='melodic')

    @classmethod
    def from_dual_regression(cls, drdir):
        """Load data from an FSL dual regression directory.
        """
        stage1 = np.loadtxt(op.join(drdir, 'dr_stage1_subject00000.txt'))
        stage2 = nib.load(op.join(drdir, 'dr_stage2_subject00000.nii.gz'))
        return cls(timeseries=stage1, maps=stage2, kind='dr')

    @classmethod
    def from_fmri_data(cls, datafile, atlas=None):
        """Take a 4D dataset and generate signals from the atlas parcels.
        """
        atlas = MIST_ATLAS_444 if atlas is None else atlas

        # Resampling target should be the image with lowest resolution.
        # Assuming that the data resolution is isotropic for now.
        atlas_res = nib.load(atlas).header['pixdim'][1]
        data_res = nib.load(datafile).header['pixdim'][1]
        resampling_target = 'data' if data_res > atlas_res else 'labels'

        masker = NiftiLabelsMasker(labels_img=atlas, standardize=True,
                                   resampling_target=resampling_target)
        signals = masker.fit_transform(datafile)
        atlasrois = atlas_roitovol(atlas, nrois=signals.shape[-1])
        return cls(timeseries=signals, maps=atlasrois, kind='atlas')


class TFM:
    """Wrapper class for FastICA and the "TFM" model data.
    """

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
        This is a separate function so we can call it from, for example,
        the RAICAR module to verify the reproducibility of the obtained
        timeseries.
        """
        sources = self.ica.fit_transform(signal)
        return sources

    def fit_transform(self, tfmdata):
        """Take a MelodicData object and unmix it.
        MelodicData does not need to be a result from Melodic at all,
        as long as its structure contains the following four elements:
        1. signal - 2D matrix of size timecourses vs ROIs.
        2. rsns - 2D matrix of size voxels x RSNs.
        3. shape - The original 3D dimensions of signal.
        4. affine - The original affine matrix of the rsns.
        """
        sources = self.unmix(tfmdata.signal)
        rsns = tfmdata.rsns
        # Use a copy so we don't mutate the internals of FastICA later
        # when reordering the TFM components.
        mixing = self.ica.mixing_.copy()

        # The TFM matrix multiplication.
        tfm = np.dot(rsns, mixing)

        # Mask outside of brain with NaN
        tfm[rsns.max(axis=-1) == 0] = np.nan

        # variance normalize *EACH COMPONENT INDIVIDUALLY.*
        # tfm -= np.nanmean(tfm, axis=0)
        tfm /= np.nanstd(tfm, axis=0)

        tfm = np.nan_to_num(tfm)

        # Because ICA orientation is arbitrary, make it such that largest value
        # is always positive.
        for spatial_map, node_weights in zip(tfm.T, mixing.T):
            if spatial_map[np.abs(spatial_map).argmax()] < 0:
                spatial_map *= -1
                node_weights *= -1

        # Now order the components according to the RMS of the mixing matrix.
        weighted_mixing = tfmdata.explainedvar * mixing
        rms = np.sum(np.square(weighted_mixing), 0)
        order = np.argsort(rms)[::-1]  # Decreasing order.

        self.reordered_mixing = mixing[:, order]
        tfm = tfm[:, order]
        sources = sources[:, order]

        return nib.Nifti1Image(np.reshape(tfm,
                                          (*tfmdata.shape, -1)),
                               tfmdata.affine), sources



def main(args):
    """Main TFM routine. Logs some information, checks user inputs, reads
    in the data, orchestrates the tICA decomposition and saves outputs to
    desired locations.
    """

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

    # Start logging. Not sure why the FileHandler doesn't work from the cluster...
    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler(out("tfm.log"), mode='w'),
                            logging.StreamHandler()
                        ])
    logging.info(f"Welcome to TFM version {__version__}")
    logging.info(sys.argv)

    # Load data
    logging.info(f"Loading data from {args.inputdir}")

    labelfile = args.labelfile if not args.no_label else None

    if os.path.isdir(args.inputdir):
        if os.path.abspath(args.inputdir).endswith('.ica'):
            tfmdata = Data.from_melodic(args.inputdir, labelfile=labelfile)
        else:
            tfmdata = Data.from_dual_regression(args.inputdir)
    else:
        tfmdata = Data.from_fmri_data(args.inputdir)

    # Parse user inputs
    n_components = min(args.n_components, len(tfmdata.signal.T))
    logging.info(f"# of signal spatial ICs is {len(tfmdata.signal.T)}")
    tolerance = args.tolerance
    max_iter = args.max_iter

    # Compute TFMs.
    logging.info("Computing TFMs")
    try_counter = 1
    algorithm = 'parallel'

    # The following loop will try unmixing the signals using a parallel
    # approach up to 5 times (in case the tICA does not converge.) In case
    # it doesn't converge in 5 attempts, it'll switch to the deflation algo,
    # which is much more stable.
    while True:
        warnings.filterwarnings("error")
        try:
            # Try using the parallel algorithm because it's faster.
            tfm_ica = TFM(n_components=n_components,
                          max_iter=max_iter,
                          tol=tolerance,
                          algorithm=algorithm,
                          random_state=np.random.randint(0, 2**32 - 1))
            tfms, sources = tfm_ica.fit_transform(tfmdata)
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

    # When using an atlas, save a heatmap of melodic unmix.
    if tfmdata.kind == 'atlas':
        f, ax = plt.subplots(figsize=plt.figaspect(1/2))
        g = heatmap(out('melodic_unmix'), ax)
        plt.tight_layout()
        f.savefig(out('melodic_unmix.png'))


def run_tfm():
    """Wrapper to be used as entry point for a command line tool.
    """

    parser = _cli_parser()
    args = parser.parse_args()
    main(args)


def _cli_parser():
    """Argument parser for run_tfm.
    """
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
