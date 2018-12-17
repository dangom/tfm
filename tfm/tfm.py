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
from typing import List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker
from sklearn.decomposition import FastICA

from . import __version__

if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('agg')
import matplotlib.pyplot as plt # noqa:E402 isort:skip
import seaborn as sns # noqa:E402 isort:skip


MIST_ROOT = op.join(op.dirname(__file__), 'mistatlas')
MIST_ATLAS_444 = op.join(MIST_ROOT, 'Parcellations/MIST_444.nii.gz')
MIST_HIERARCHY = op.join(MIST_ROOT, 'Hierarchy/MIST_PARCEL_ORDER.csv')


def mist_parcel_info(nrois: int = 12) -> str:
    """Return the filename of the csv with parcel information (label, size,
    etc.) from the MIST atlas with a number of parcels equal to `nrois`.
    """
    return op.join(MIST_ROOT, f'Parcel_Information/MIST_{nrois}.csv')


# TODO: Perhaps return a dataframe with original roi, target roi and labels
# makes more sense.
def atlas_parcel_labels(nrois: int,
                        target_res: int = 12) -> Tuple[List, List, List]:
    """This function takes two integer numbers, nrois and target_res. nrois
    indicates the number of MIST parcels used for an atlas based tICA
    decomposition, and target_res indicates the number of MIST parcels (higher
    in the MIST hierarchy) that label each of the original nrois.

    The idea is to group ROIs into higher level networks only for labelling and
    summarizing.
    """
    hierarchy: pd.DataFrame = pd.read_csv(MIST_HIERARCHY, delimiter=',')
    roi_ids: np.array = np.unique(hierarchy[f's{nrois}'].values)

    rois_parent: list = [
        np.unique(hierarchy[f's{target_res}'][hierarchy[f's{nrois}'] == x])
        for x in roi_ids]

    # Because all values are the same, take the first one. Use tolist() to get
    # a single flat list in the end.
    rois_parent: list = [x.tolist()[0] for x in rois_parent]
    labels: pd.DataFrame = pd.read_csv(mist_parcel_info(target_res),
                                       delimiter=';')

    parent_names: list = [
        labels[labels['roi'] == x]['name'].values.tolist()[0]
        for x in rois_parent]

    parent_labels: list = [
        labels[labels['roi'] == x]['label'].values.tolist()[0]
        for x in rois_parent]

    return rois_parent, parent_names, parent_labels


def labeled_unmix(unmix: str) -> pd.DataFrame:
    """unmix is the filename of a melodic_unmix matrix, i.e., the "core" tICA
    mixing matrix. This routine takes the filename and returns a pandas
    DataFrame with columns: label, name, roi (numbered parcels), tfm (numbered
    tfms), coefficient (roi x tfm entry), abs_coefficient
    (np.abs(coefficient)). """
    unmix_df: pd.DataFrame = pd.DataFrame(np.loadtxt(unmix))
    rois_parent, names, labels = atlas_parcel_labels(len(unmix_df))
    unmix_df['name'] = names
    unmix_df['label'] = labels
    unmix_df = unmix_df.reset_index().melt(['index', 'label', 'name'])
    unmix_df = unmix_df.rename(columns={'index': 'roi',
                                        'variable': 'tfm',
                                        'value': 'coefficient'})
    unmix_df['abs_coefficient'] = unmix_df['coefficient'].abs()
    return unmix_df


def data_summary(filename_or_matrix: Union[str, np.array],
                 target_res: int = 12,
                 normalise: bool = True,
                 absolute_vals: bool = True) -> pd.DataFrame:
    """Return a DataFrame with the core matrix summarized by
    networks. Values are given by % contribution to each tfm.
    """
    if not isinstance(filename_or_matrix, str):
        mat = filename_or_matrix
    else:
        mat = np.loadtxt(filename_or_matrix)

    data_raw = pd.DataFrame(mat)
    if absolute_vals:
        data_raw = np.abs(data_raw)
    _, parent_names, _ = atlas_parcel_labels(len(data_raw), target_res)
    data_raw['name'] = parent_names
    data = data_raw.groupby('name').aggregate(sum)

    # Normalise TFMs such that contributions of all networks sum to 100%
    if normalise:
        for column in data:
            data[column] = 100*data[column]/np.sum(np.abs(data[column]))

    # Replace '_' for ' ' in the network labels.
    data.index = pd.Index(np.array(list(map(lambda x: x.replace('_', ' '),
                                            data.index.values)),
                                   dtype='object'))

    return data


def sort_by_visual(matrix: np.array) -> np.array:
    summary = data_summary(matrix)
    order = summary.sort_values('VISUAL NETWORK', axis=1).columns
    return order.values.astype(int)[::-1]


def heatmap(data: np.array, ax, **kwargs):
    """Plot the core TFM mixing matrix as a heatmap, which ROIs contributions
    aggregated by the 12 MIST functional networks.
    """
    g = sns.heatmap(data,
                    ax=ax,
                    **kwargs)

    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    return g


def correlate(a: np.array, b: np.array) -> np.array:
    """Fast numpy Row-wise Corr. Coefficients for 2D arrays.
    See benchmarks at https://stackoverflow.com/a/30143754/3568092
    :param a: 2D array
    :param b: 2D array
    :returns: Corr Matrix between rows of a with rows of b
    :rtype: 2D array
    """
    # Center vectors by subtracting row mean.
    a_centered = a - a.mean(1)[:, None]
    b_centered = b - b.mean(1)[:, None]

    # Sum of squares across rows.
    a_sos = (a_centered ** 2).sum(1)
    b_sos = (b_centered ** 2).sum(1)

    norm_factors = np.sqrt(np.dot(a_sos[:, None], b_sos[None]))
    # Finally get corr coeff
    return np.dot(a_centered, b_centered.T) / norm_factors


def correlation_with_confounds(signal: np.array,
                               confounds: pd.DataFrame) -> pd.DataFrame:

    corrmat = correlate(signal.T, np.nan_to_num(confounds.values.T))
    corrdf = pd.DataFrame(np.abs(corrmat), columns=confounds.columns).T

    return corrdf


def double_heatmap(corrdf1: np.array, corrdf2: np.array) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=plt.figaspect(1/2))
    heatmap(corrdf1, vmin=0.35, vmax=0.8, ax=ax1,
            cbar=False, yticklabels=1,
            xticklabels=range(1, corrdf1.shape[1] + 1))
    ax1.set_title('sICA Timeseries')
    ax1.set_xlabel('Signal Component Index')
    heatmap(corrdf2, vmin=0.35, vmax=0.8, ax=ax2,
            yticklabels=False,
            xticklabels=range(1, corrdf2.shape[1] + 1))
    ax2.set_title('tICA Timeseries')
    ax2.set_xlabel('TFM Index')
    plt.tight_layout()
    return fig


def parse_melodic_labelfile(labelfile: str) -> List[int]:
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


def atlas_roitovol(atlas: str, nrois: int) -> nib.Nifti1Image:
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
    def __init__(self,
                 timeseries: np.array,
                 maps: nib.Nifti1Image,
                 kind: Optional[str] = None,
                 confounds: Optional[str] = None,
                 decimate: Optional[int] = None,
                 skipfirst: Optional[int] = None,
                 skiplast: Optional[int] = None) -> None:
        """Timeseries is a numpy array.
        Maps is a nibabel Nifti1Image object.
        """
        assert timeseries.shape[-1] == maps.shape[-1]
        if skiplast is not None:
            skiplast *= -1
        self.timeseries = timeseries[skipfirst:skiplast:decimate]
        self.maps = maps

        # here for compatibility with previous versions
        self.shape = maps.shape[:-1]
        self.affine = maps.affine
        self.explainedvar = 1

        self.kind = kind

        # Experimental support for fmriprep confounds
        if confounds is not None:
            cofs = pd.read_csv(confounds, delimiter='\t')
            self.confounds = cofs[skipfirst:skiplast:decimate]
        else:
            self.confounds = None

    @property
    def rsns(self) -> np.array:
        n_parcels = self.maps.shape[-1]
        data = np.reshape(self.maps.get_data(), (-1, n_parcels))
        return data

    @property
    def signal(self) -> np.array:
        return self.timeseries

    @classmethod
    def from_melodic(cls,
                     icadir: str,
                     labelfile: Optional[str] = None,
                     confounds: Optional[str] = None,
                     **kwargs):
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

        return cls(timeseries=melodic_mix, maps=melodic_oic, kind='melodic',
                   confounds=confounds, **kwargs)

    @classmethod
    def from_dual_regression(cls,
                             drdir: str,
                             confounds: Optional[str] = None,
                             **kwargs):
        """Load data from an FSL dual regression directory.
        """
        stage1 = np.loadtxt(op.join(drdir, 'dr_stage1_subject00000.txt'))
        stage2 = nib.load(op.join(drdir, 'dr_stage2_subject00000.nii.gz'))
        return cls(timeseries=stage1, maps=stage2, kind='dr',
                   confounds=confounds, **kwargs)

    @classmethod
    def from_fmri_data(cls,
                       datafile: str,
                       atlas: Optional[str] = None,
                       confounds: Optional[str] = None,
                       **kwargs):
        """Take a 4D dataset and generate signals from the atlas parcels.
        """
        if atlas is None:
            atlas = MIST_ATLAS_444
            kind = 'atlas'
        else:
            kind = 'atlas_custom'

        # Resampling target should be the image with lowest resolution.
        # Assuming that the data resolution is isotropic for now.
        atlas_res = nib.load(atlas).header['pixdim'][1]
        data_res = nib.load(datafile).header['pixdim'][1]
        resampling_target = 'data' if data_res > atlas_res else 'labels'

        masker = NiftiLabelsMasker(labels_img=atlas, standardize=True,
                                   resampling_target=resampling_target)
        signals = masker.fit_transform(datafile)
        atlasrois = atlas_roitovol(atlas, nrois=signals.shape[-1])
        return cls(timeseries=signals, maps=atlasrois, kind=kind,
                   confounds=confounds, **kwargs)


class TFM:
    """Wrapper class for FastICA and the "TFM" model data.
    """

    # Deflation takes much longer, but converges more often.
    def __init__(self,
                 n_components: int = 30,
                 max_iter: int = 20_000,
                 tol: float = 0.00001,
                 fun='logcosh', algorithm='parallel', random_state=None,
                 demean_tfms: bool = True):

        self.ica = FastICA(max_iter=max_iter, tol=tol,
                           n_components=n_components,
                           fun=fun,
                           algorithm=algorithm,
                           random_state=random_state)

        self.demean_tfms = demean_tfms

    def unmix(self, signal: np.array) -> np.array:
        """Call FastICA on the signal components.
        This is a separate function so we can call it from, for example,
        the RAICAR module to verify the reproducibility of the obtained
        timeseries.
        """
        sources = self.ica.fit_transform(signal)
        return sources

    def fit_transform(self,
                      tfmdata: Data) -> Tuple[nib.Nifti1Image, np.array]:
        """Take a Data object and unmix it.
        Data does not need to be a result from Melodic at all,
        as long as its structure contains the following four elements:
        1. signal - 2D matrix of size timecourses vs ROIs.
        2. rsns - 2D matrix of size voxels x RSNs.
        3. shape - The original 3D dimensions of signal.
        4. affine - The original affine matrix of the rsns.
        """
        sources = self.unmix(tfmdata.signal)  # # timepoints X # tfms
        rsns = tfmdata.rsns
        # Use a copy so we don't mutate the internals of FastICA later
        # when reordering the TFM components.
        mixing = self.ica.mixing_.copy()  # # ROIs X # tfms

        # The TFM matrix multiplication.
        tfm = np.dot(rsns, mixing)  # # voxels X # tfms

        # Mask outside of brain with NaN
        tfm[rsns.max(axis=-1) == 0] = np.nan

        # variance normalize *EACH COMPONENT INDIVIDUALLY.*
        if self.demean_tfms:
            tfm -= np.nanmean(tfm, axis=0)
        tfm /= np.nanstd(tfm, axis=0)

        tfm = np.nan_to_num(tfm)

        # Because ICA orientation is arbitrary, make it such that largest value
        # is always positive. Also flip the timeseries.
        for spatial_map, ts in zip(tfm.T, sources.T):
            if spatial_map[np.abs(spatial_map).argmax()] < 0:
                spatial_map *= -1
                ts *= -1

        # Now order the components according to the RMS of the mixing matrix.
        # weighted_mixing = tfmdata.explainedvar * mixing
        # rms = np.sum(np.square(weighted_mixing), 0)
        # order = np.argsort(rms)[::-1]  # Decreasing order.
        if tfmdata.kind == 'atlas':
            order = sort_by_visual(mixing)
            self.reordered_mixing = mixing[:, order]
            tfm = tfm[:, order]
            sources = sources[:, order]
        else:
            self.reordered_mixing = mixing

        return nib.Nifti1Image(np.reshape(tfm,
                                          (*tfmdata.shape, -1)),
                               tfmdata.affine), sources


def _check_dirs(args) -> str:
    """Check that input and output directories are OK.
    If the output directory is not an absolute directory, then
    consider it to be relative to the input directory.
    Return that.
    """

    errmsg = f'Input {args.inputdir} does not exist or is not accessible.'
    assert os.path.exists(args.inputdir), errmsg

    if os.path.isabs(args.outputdir):
        outdir = args.outputdir
    else:
        outdir = os.path.join(os.path.dirname(args.inputdir), args.outputdir)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    else:
        if not args.force:
            errmsg = 'Already existing files in output directory.'
            assert os.listdir(outdir) == "", errmsg

    return outdir


def _data_loader(args) -> Data:
    """Small wrapper to load the data either from melodic, dual_regression
    or atlas.
    """
    labelfile = args.labelfile if not args.no_label else None

    skipvols = {'skipfirst': args.skipfirst,
                'skiplast': args.skiplast,
                'decimate': args.decimate}

    if os.path.isdir(args.inputdir):
        if os.path.abspath(args.inputdir).endswith('.ica'):
            tfmdata = Data.from_melodic(args.inputdir, labelfile=labelfile,
                                        confounds=args.confounds, **skipvols)
        else:
            tfmdata = Data.from_dual_regression(args.inputdir,
                                                confounds=args.confounds,
                                                **skipvols)
    else:
        tfmdata = Data.from_fmri_data(args.inputdir,
                                      atlas=args.atlas,
                                      confounds=args.confounds,
                                      **skipvols)

    return tfmdata


def main(args) -> None:
    """Main TFM routine. Logs some information, checks user inputs, reads
    in the data, orchestrates the tICA decomposition and saves outputs to
    desired locations.
    """

    # Note, this function has side-effects.
    outdir = _check_dirs(args)

    def out(name: str) -> str:
        return os.path.join(outdir, name)

    # Start log. Not sure why the FileHandler doesn't work on the cluster...
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

    tfmdata = _data_loader(args)

    # Parse user inputs
    tolerance, max_iter = args.tolerance, args.max_iter

    n_components = min(args.n_components, len(tfmdata.signal.T))
    logging.info(f"# of signal spatial ICs is {len(tfmdata.signal.T)}")

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
                          random_state=np.random.randint(0, 2**32 - 1),
                          demean_tfms=args.skipdemean)
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
    logging.info(f"TFM demeaning set to: {tfm_ica.demean_tfms}")

    # Save outputs
    logging.info(f"Saving outputs to directory {outdir}")
    tfms.to_filename(out('melodic_IC.nii.gz'))
    np.savetxt(out('melodic_unmix'), tfm_ica.reordered_mixing,
               delimiter='  ', fmt='%.6f')
    np.savetxt(out('melodic_mix'), sources, delimiter='  ', fmt='%.6f')
    np.savetxt(out('melodic_FTmix'), np.abs(np.fft.rfft(sources, axis=0)),
               delimiter='  ', fmt='%.6f')

    if tfmdata.confounds is not None:
        # cofs = pd.read_csv(tfmdata.confounds, delimiter='\t')
        dfsignal = correlation_with_confounds(tfmdata.signal,
                                              tfmdata.confounds)
        dfsignal.to_csv(out('signal_correlation_to_confounds.csv'))
        dftfm = correlation_with_confounds(sources, tfmdata.confounds)
        contamination = dftfm.max(axis=0).values
        dftfm.to_csv(out('tfm_correlation_to_confounds.csv'))
        fig = double_heatmap(dfsignal, dftfm)
        # TODO Why a different filename here? Fix this in future release.
        fig.savefig(out('correlation_with_confounds.png'))

    if tfmdata.kind == 'atlas':
        # When using an atlas, save a heatmap of melodic unmix.
        df = data_summary(out('melodic_unmix'))
        df.to_csv(out('network_contributions.csv'))
        f, ax = plt.subplots(figsize=plt.figaspect(1/2))
        heatmap(df, ax, vmin=5, vmax=20, yticklabels=1,
                annot=True, fmt=".1f", linewidth=.5,
                xticklabels=range(1, df.shape[1] + 1))
        ax.set_xlabel('TFM Index')
        # Should simplify the logic of these if calls... eventually.
        if tfmdata.confounds is not None:
            contamination = dftfm.max(axis=0).values.tolist()
            # Green for clean, red for contaminated (interp over yellow.)
            # colors = [(min(1, 2 * x),
            #            min(1, 2 * (1 - x)), 0) for x in contamination]
            colors = ['black' if x < 0.5 else 'red' for x in contamination]
            for label, color in zip(ax.get_xticklabels(), colors):
                label.set_color(color)
                label.set_weight('bold')
        plt.tight_layout()
        f.savefig(out('melodic_unmix.png'))
        # Also save the summary with MIST 64 instead of only MIST 12.
        df_64 = data_summary(out('melodic_unmix'), target_res=64)
        df_64.to_csv(out('network_contributions_64.csv'))


def run_tfm() -> None:
    """Wrapper to be used as entry point for a command line tool.
    """

    args = _cli_parser().parse_args()
    main(args)


def run_correlation_with_confounds() -> None:

    args = _cli_parser().parse_args()

    # Note, this function may create a directory (side-effects)
    outdir = _check_dirs(args)

    def out(name: str) -> str:
        return os.path.join(outdir, name)

    cofs = pd.read_csv(args.confounds, delimiter='\t')
    dftfm = correlation_with_confounds(np.loadtxt(out('melodic_mix')),
                                       cofs)
    dftfm.to_csv(out('tfm_correlation_to_confounds.csv'))


def run_summary_tfms() -> None:

    args = _cli_parser().parse_args()

    # Note, this function may create a directory (side-effects)
    outdir = _check_dirs(args)

    def out(name: str) -> str:
        return os.path.join(outdir, name)

    df = data_summary(out('melodic_unmix'), target_res=20,
                      normalise=True,
                      absolute_vals=False)
    df.to_csv(out('network_contributions_raw.csv'))
    f, ax = plt.subplots(figsize=plt.figaspect(1/2))
    heatmap(df, ax, vmin=-7, vmax=7, yticklabels=1,
            annot=True, fmt=".0f", linewidth=.4)
    ax.set_xlabel('TFM Index')
    ax.set_xticklabels(range(1, df.shape[1] + 1),
                       rotation=0)  # ha="right"
    if op.exists(out('tfm_correlation_to_confounds.csv')):
        cofs = pd.read_csv('tfm_correlation_to_confounds.csv',
                           index_col=0)
        contamination = cofs.max(axis=0).values.tolist()
        # colors = [(min(1, 2 * x),
        #            min(1, 2 * (1 - x)), 0) for x in contamination]
        colors = ['black' if x < 0.5 else 'red' for x in contamination]
        for label, color in zip(ax.get_xticklabels(), colors):
            label.set_color(color)
            label.set_weight('bold')
    plt.tight_layout()
    f.savefig(out('melodic_unmix_raw.png'))
    # Also save the summary with MIST 64 instead of only MIST 12.
    df_64 = data_summary(out('melodic_unmix'), target_res=64,
                         normalise=False,
                         absolute_vals=False)
    df_64.to_csv(out('network_contributions_64_raw.csv'))


def _cli_parser() -> argparse.ArgumentParser:
    """Argument parser for run_tfm.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    # output directory
    parser.add_argument('inputdir', type=str,
                        help=('Input directory with melodic data,'
                              'or input file for atlas-based tICA.'))

    parser.add_argument('-o', '--outputdir', type=str, default='tfm.ica',
                        help=('Directory where results will be stored.'
                              ' Def: tfm.ica'))

    parser.add_argument('-l', '--labelfile', type=str,
                        default='hand_classification.txt',
                        help=('Name of classification file.'
                              ' Default hand_classification.txt'))

    parser.add_argument('--confounds', type=str,
                        default=None,
                        help=('Name of confounds file '
                              '(tested with fmriprep confound files).'))

    parser.add_argument('--n_components', type=int, default=None,
                        help=('Number of components to extract'
                              ' from tICA decomposition'))

    parser.add_argument('--decimate', type=int, default=None,
                        help=('Take only every other Nth timepoint'
                              ' from the original signals'))

    parser.add_argument('--skipfirst', type=int, default=None,
                        help=('Skip the first N volumes'))

    parser.add_argument('--skiplast', type=int, default=None,
                        help=('Skip the last N volumes'))

    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='ICA tolerance. Default 1e-4')

    parser.add_argument('--max_iter', type=int, default=10_000,
                        help='ICA max number of iterations')

    parser.add_argument('--force', action='store_true',
                        help='Overwrite files.')

    parser.add_argument('--atlas', type=str, default=None,
                        help=('Atlas to use for generating TFMs,'
                              'in case of atlas-based approach. Default: MIST 444'))

    parser.add_argument('--no-label', action='store_true',
                        help='Consider all components to be signals.')

    parser.add_argument('--skipdemean', action='store_false',
                        help='Do not demean TFM maps.')

    return parser
