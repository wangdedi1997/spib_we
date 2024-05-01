import numpy as np
import torch
from westpa.core.binning.assign import BinMapper
from westpa.core.binning import RectilinearBinMapper, VoronoiBinMapper

index_dtype = np.uint16
coord_dtype = np.float32
# define constant EPS
EPS = np.finfo(np.float32).eps

def dfunc(p, centers, *dfargs, **dfkwargs):
    d = np.sqrt(np.square(centers - p).sum(axis=-1))
    return d

def generate_initial_labels(input_data, initial_state_num, batch_size, SPIB_model_folder):
    '''
        Generate the initial state labels for SPIB training by uniformly discretizing the expert-based CVs using
        regular space clustering.
            Args:
                input_data: ndarray containing (n,l,d)-shaped float data
                initial_state_num: the number of initial states, integer greater than 0 required
                batch_size: int
                SPIB_model_folder: str, the path to save the generated initial state labels

            Returns:
                output: ndarray containing (n,l)-shaped integer data
    '''

    # take the user-defined pcoord for regular space clustering
    coord_len = input_data.shape[1]
    input_data = input_data.reshape((-1, input_data.shape[-1]))

    # standardize the input data for regular space clustering
    input_data_range = input_data.max(axis=0) - input_data.min(axis=0)

    # Uniformly discretize the space to get the desired number of clusters using regular space clustering.
    cluster_centers = UniformClustering(input_data/input_data_range, cluster_num=initial_state_num,
                                        batch_size=batch_size)

    bin_mapper = VoronoiBinMapper(dfunc, np.require(cluster_centers*input_data_range, dtype=coord_dtype))

    # get SPIB pcoord and state label
    initial_labels = []
    for i in range(0, len(input_data), batch_size):
        batch_inputs = input_data[i:i + batch_size]
        label = bin_mapper.assign(batch_inputs)

        initial_labels += [label]

    # get initial state labels
    initial_labels = np.concatenate(initial_labels, 0)

    # remove empty assignments
    n_state = initial_labels.max() + 1

    output = np.zeros_like(initial_labels, dtype=int)
    idx = 0
    for i in range(n_state):
        indices = (initial_labels == i)
        if indices.sum() == 0:
            continue
        output[indices] = idx
        idx += 1

    output = np.reshape(output, (-1, coord_len,))
    np.save(SPIB_model_folder + '/initial_labels.npy', output)
    return output


def UniformSpaseGridding(input_data, h):
    '''
        Uniformly grid the space using dmin.
            Args:
                input_data: ndarray containing (n,d)-shaped float data
                h: float, the bin size

            Returns:
                grid_list: d lists of bin boundaries
    '''

    grid_list = [['-inf',]+[i for i in np.arange(input_data[:, idx].min(axis=0), input_data[:, idx].max(axis=0), h)] +
                 ['inf',] for idx in range(input_data.shape[1])]

    return grid_list


def UniformClustering(input_data, cluster_num, tol=0.1, batch_size=128, max_trials=5, return_dmin=False):
    '''
        Uniformly discretize the space using regular space clustering.
            Args:
                input_data: ndarray containing (n,d)-shaped float data
                cluster_num: the desired number of cluster centers expected to obtain, integer greater than 0 required
                tol: the acceptable difference between the expected number of cluster centers and the actual number
                    of cluster centers

            Returns:
                center_list: ndarray containing the cluster centers
    '''

    dim = input_data.shape[-1]

    initial_min_dist = np.power(np.prod(input_data.max(axis=0)-
                                        input_data.min(axis=0))/cluster_num, 1.0/dim)

    # use half of initial_min_dist as dmin for regular space clustering
    min_dist = initial_min_dist / 2
    cluster_centers = RegSpaceClustering(input_data, min_dist, max_centers=cluster_num*10,
                                         batch_size=batch_size)

    current_cluster_num = cluster_centers.shape[0]
    trial_num = 1

    # keep refining the min_dist until the difference is acceptable
    while 1.0*np.abs(cluster_num - current_cluster_num)/cluster_num > tol and trial_num < max_trials:
        min_dist = np.power(current_cluster_num*(min_dist)**dim/cluster_num, 1.0/dim)
        cluster_centers = RegSpaceClustering(input_data, min_dist, max_centers=cluster_num * 10,
                                             batch_size=batch_size)
        current_cluster_num = cluster_centers.shape[0]

        trial_num += 1

    if return_dmin:
        return cluster_centers, min_dist
    else:
        return cluster_centers

def UniformGridding(input_data, bin_num, tol=0.1, max_trials=5):
    '''
        Uniformly discretize the space using a rectilinear grid.
            Args:
                input_data: ndarray containing (n,d)-shaped float data
                bin_num: the desired number of occupied bins, integer greater than 0 required
                tol: the acceptable difference between the expected number of cluster centers and the actual number
                    of cluster centers

            Returns:
                grid_list: d lists of bin boundaries
    '''

    dim = input_data.shape[-1]

    initial_h = np.power(np.prod(input_data.max(axis=0) - input_data.min(axis=0))/bin_num, 1.0/dim)

    # use half of initial_h as h for uniformly gridding
    h = initial_h / 2
    grid_list = UniformSpaseGridding(input_data, h)

    bin_mapper = RectilinearBinMapper(grid_list)

    current_bin_num = len(np.unique(bin_mapper.assign(input_data)))
    trial_num = 1

    # keep refining the h until the difference is acceptable
    while 1.0*np.abs(bin_num - current_bin_num)/bin_num > tol and trial_num < max_trials:
        h = np.power(current_bin_num * (h)**dim/bin_num, 1.0/dim)

        grid_list = UniformSpaseGridding(input_data, h)
        bin_mapper = RectilinearBinMapper(grid_list)
        current_bin_num = len(np.unique(bin_mapper.assign(input_data)))

        trial_num += 1

    return grid_list

def RegSpaceClustering(input_data, min_dist, max_centers=200, batch_size=256):
    '''
    Regular space clustering.
        Args:
            input_data: ndarray containing (n,d)-shaped float data
            max_centers: the maximum number of cluster centers to be determined, integer greater than 0 required
            min_dist: the minimal distances between cluster centers

        Returns:
            center_list: ndarray containing the cluster centers
    '''

    num_observations, d = input_data.shape

    p = np.random.permutation(num_observations)
    data = input_data[p]

    center_list = data[0:1, :].copy()

    i = 1
    while i < num_observations:
        x_active = data[i:i+batch_size, :]
        distances = np.sqrt((np.square(np.expand_dims(center_list,0) - np.expand_dims(x_active,1))).sum(axis=-1))
        indice = np.nonzero(np.all(distances > min_dist, axis=-1))[0]
        if len(indice) > 0:
            # the first element will be used
            center_list = np.concatenate((center_list, x_active[indice[0]].reshape(1, d)), 0)
            i += indice[0]
        else:
            i += batch_size

        if len(center_list) >= max_centers:
            print("Exceed the maximum number of cluster centers!\n")
            print("Please increase dmin!\n")
            break

    return center_list

class SPIBBinMapper(BinMapper):
    """


    A hybrid bin mapper to A hybrid bin mapper to wrap the expert-based bins and SPIB learned bins for SPIB-WE.

    Parameters
    ----------
    original_bin_mapper : BinMapper
        The user-specified bin mapper for expert-based CVs.
    pcoord_ndim: int, default=3
        Dimension of progress coordinate, which should be the sum of the dimensions of expert-based CVs and SPIB-learned CVs
    SPIB: SPIB model
        The SPIB model obtained from training.
    spib_ndim: int, default=2
        Dimension of SPIB learned CVs
    spib_grid_lists : spib_ndim lists of bin boundaries
    expected_spib_bin_num: int, default=100
        The expected number of occupied SPIB bins to be determined in the SPIB space.
    tol : float, default=0.2
        The acceptable relative difference between the expected number of occupied SPIB bins and the actual number
        of occupied SPIB bins.
    """

    def __init__(self, original_bin_mapper, pcoord_ndim=3, SPIB=None, spib_ndim=2, spib_grid_lists=None,
                 expected_spib_bin_num=100, tol=0.2, args=None, kwargs=None):
        self.original_bin_mapper = original_bin_mapper
        self.original_nbins = original_bin_mapper.nbins

        self.original_bin_population = np.zeros(self.original_nbins, dtype=int)
        self.under_sampled_original_bin = np.zeros(self.original_nbins, dtype=bool)
        self.expected_spib_bin_num = expected_spib_bin_num
        self.tol = tol

        self.pcoord_ndim = pcoord_ndim
        self.SPIB = SPIB
        self.spib_ndim = spib_ndim

        if self.SPIB is None:
            self.spib_nbins = 1
            self.nstates = 1
        else:
            self.nstates = SPIB.output_dim
            assert spib_ndim == SPIB.z_dim
            if spib_grid_lists is None:
                self.spib_bin_mapper = None
                self.spib_nbins = self.nstates
            else:
                self.spib_bin_mapper = RectilinearBinMapper(spib_grid_lists)
                self.spib_nbins = self.spib_bin_mapper.nbins

        self.spib_bin_population = np.zeros(self.spib_nbins, dtype=int)

        # set more bins to avoid the overflow
        self.nbins = int(self.original_nbins + self.spib_nbins)

        self.labels = ['{:d}'.format(ibin) for ibin in range(self.nbins)]

        self.args = args or ()
        self.kwargs = kwargs or {}

    def assign(self, coords, mask=None, output=None):
        try:
            passed_coord_dtype = coords.dtype
        except AttributeError:
            coords = np.require(coords, dtype=coord_dtype)
        else:
            if passed_coord_dtype != coord_dtype:
                coords = np.require(coords, dtype=coord_dtype)

        if coords.ndim != 2:
            raise TypeError('coords must be 2-dimensional')
        if mask is None:
            mask = np.ones((len(coords),), dtype=np.bool_)
        elif len(mask) != len(coords):
            raise TypeError('mask [shape {}] has different length than coords [shape {}]'.format(mask.shape, coords.shape))

        if output is None:
            output = np.empty((len(coords),), dtype=index_dtype)
        elif len(output) != len(coords):
            raise TypeError('output has different length than coords')

        original_ibin = self.original_bin_mapper.assign(coords)

        if self.SPIB is not None:
            if self.spib_bin_mapper is not None:
                spib_ibin = self.spib_bin_mapper.assign(coords[:, -self.spib_ndim:])
            else:
                spib_ibin = self.get_SPIB_state(coords[:, -self.spib_ndim:])

            # check original_bin_occupancy or spib_bin_occupancy
            under_sampled_original_ibin = self.under_sampled_original_bin[original_ibin]
            original_ibin_population = self.original_bin_population[original_ibin]
            spib_ibin_population = self.spib_bin_population[spib_ibin]

            # if original_ibin is well-sampled or spib_ibin population <= original_ibin population:
            #   assign sample i to spib_ibin + self.original_nbins
            # else:
            #   assign sample i to original_ibin
            output = np.where(np.logical_and(under_sampled_original_ibin, original_ibin_population < spib_ibin_population),
                              original_ibin, spib_ibin + self.original_nbins)

        else:
            output = original_ibin

        return output

    def update_bin_population(self, current_coords):

        # initialize bin
        self.original_bin_population = np.zeros(self.original_nbins, dtype=int)

        # Pull the coordinates from the current iteration
        coords = current_coords.reshape(-1, self.pcoord_ndim)

        # get original bin population
        original_ibin = self.original_bin_mapper.assign(coords)

        # count the original bin population
        for ibin in original_ibin:
            self.original_bin_population[ibin] += 1

        # get spib bin population
        if self.SPIB is not None:
            if self.spib_bin_mapper is not None:
                spib_ibin = self.spib_bin_mapper.assign(coords[:, -self.spib_ndim:])

                occupied_spib_nbin = len(np.unique(spib_ibin))

                if 1.0 * np.abs(
                    occupied_spib_nbin - self.expected_spib_bin_num) / self.expected_spib_bin_num > self.tol:
                    self.update_spib_bin_mapper(coords[:, -self.spib_ndim:])
                    # get new ibin
                    spib_ibin = self.spib_bin_mapper.assign(coords[:, -self.spib_ndim:])

            else:
                spib_ibin = self.get_SPIB_state(coords[:, -self.spib_ndim:])

            self.spib_bin_population = np.zeros(self.spib_nbins, dtype=int)

            # count the original bin population
            for ibin in spib_ibin:
                self.spib_bin_population[ibin] += 1

    def update_spib_bin_mapper(self, current_coords):

        # Uniformly discretize the space to get the desired number of clusters using regular space clustering.
        spib_grid_lists = UniformGridding(current_coords, bin_num=self.expected_spib_bin_num)

        self.spib_bin_mapper = RectilinearBinMapper(spib_grid_lists)
        self.spib_nbins = self.spib_bin_mapper.nbins

        self.spib_bin_population = np.zeros(self.spib_nbins, dtype=int)

        # set more bins to avoid the overflow
        self.nbins = int(self.original_nbins + self.spib_nbins)

        self.labels = ['{:d}'.format(ibin) for ibin in range(self.nbins)]


    def update_under_sampled_bins(self, current_coords):

        # initialize bin
        original_bin_population = np.zeros(self.original_nbins, dtype=int)

        # Pull the coordinates from the current iteration
        coords = current_coords.reshape(-1, self.pcoord_ndim)

        # get original bin population
        original_ibin = self.original_bin_mapper.assign(coords)

        totol_population = len(original_ibin)
        # count the original bin population
        for ibin in original_ibin:
            original_bin_population[ibin] += 1

        occupied_original_nbins = np.sum(original_bin_population > 0)

        # identify the under-sampled region
        # population is smaller than the average population
        self.under_sampled_original_bin = (
                original_bin_population < totol_population / occupied_original_nbins).astype(bool)


    @torch.no_grad()
    def get_SPIB_prediction(self, spib_coord):
        if self.SPIB is None:
            return np.zeros((len(spib_coord),)).astype(int)
        else:
            z = torch.from_numpy(spib_coord).float().to(self.SPIB.device)
            log_prediction = self.SPIB.decode(z)
            output = log_prediction.exp()
            return output.cpu().data.numpy()

    @torch.no_grad()
    def get_SPIB_state(self, spib_coord):
        if self.SPIB is None:
            return np.zeros((len(spib_coord),)).astype(int)
        else:
            z = torch.from_numpy(spib_coord).float().to(self.SPIB.device)
            log_prediction = self.SPIB.decode(z)
            output = log_prediction.exp().argmax(1)
            return output.cpu().data.numpy()

    @torch.no_grad()
    def get_SPIB_rcoord(self, input):
        if self.SPIB is None:
            return np.zeros((len(input),))
        else:
            input = torch.from_numpy(input).float().to(self.SPIB.device)
            spib_coord, _ = self.SPIB.encode(input)
            return spib_coord.cpu().data.numpy()



