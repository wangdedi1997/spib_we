import numpy as np
import os
import westpa
import torch
import random
import h5py
import logging
import time
from spib.spib import SPIB
from spib.utils import prepare_data, DataNormalize
from spib_we.spib_binning import SPIBBinMapper, generate_initial_labels

log = logging.getLogger(__name__)

# define constant EPS
EPS = np.finfo(np.float32).eps

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    else:
        return False

class SPIBDriver:
    """
    WESTPA plugin to automatically train SPIB model for CV construction. It also supports the use of expert-based CVs to
    encourage the exploration of new regions.

    Can be used by including the following entries in your west.cfg:
        west:
            plugins:
              - plugin: spib_we.spib_driver.SPIBDriver
                initial_iter: int, number of initial WE iterations before starting SPIB.
                update_interval: int, number of interval WE iterations between SPIB updates.
                collect_last_n_iter: int, number of iterations of data used for SPIB training.
                exclude_initial_n_iter: int, number of initial iterations of data excluded for SPIB training.
                spib_ndim: int, dimensionality of bottleneck.
                lagtime: int, time delay delta t in terms of # of minimal time resolution of the trajectory data
                encoder_type: str, Encoder type (Linear or Nonlinear)
                neuron_num1: int, number of nodes in each hidden layer of the encoder
                neuron_num2: int, number of nodes in each hidden layer of the decoder
                batch_size: int, batch size
                tolerance: float, tolerance of loss change for measuring the convergence of the training
                patience: int, Number of epochs with the change of the state population smaller than the threshold after
                    which this iteration of the training finishes
                refinements: int, number of refinements
                learning_rate: float, learning rate of Adam optimizer
                beta: Hyperparameter beta
                initial_state_num: the approximate number of regular space clusters for initial state labels,
                    integer greater than 0 required
                expected_occupied_bin_num: the expected number of occupied bins to be determined in the SPIB space,
                    integer greater than 0 required
                enable_weights: Bool, whether to weight input data for training.
                enable_data_transform: Bool, whether to normalize input data before input the neural networks.
    """

    def __init__(self, sim_manager, plugin_config):
        westpa.rc.pstatus("Initializing optimization plugin")

        if not sim_manager.work_manager.is_master:
            westpa.rc.pstatus("Not running on the master process, skipping")
            return

        self.data_manager = sim_manager.data_manager
        self.sim_manager = sim_manager
        self.system = sim_manager.system
        self.we_driver = westpa.rc.get_we_driver()
        self.propagator = westpa.rc.get_propagator()

        self.plugin_config = plugin_config

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.default_device = torch.device("cpu")

        # Number of data points per iteration
        self.pcoord_len = int(westpa.rc.config.get(
            ["west", "system", "system_options", "pcoord_len"]
        ))

        # get the dimension of pcoord (original CV dim + SPIB dim)
        self.pcoord_ndim = int(westpa.rc.config.get(
            ["west", "system", "system_options", "pcoord_ndim"]
        ))

        self.initial_iter = int(plugin_config.get("initial_iter", 1))
        self.update_interval = int(plugin_config.get("update_interval", 1))
        self.collect_last_n_iter = int(plugin_config.get("collect_last_n_iter", 1))
        self.exclude_initial_n_iter = int(plugin_config.get("exclude_initial_n_iter", 0))
        self.spib_ndim = int(plugin_config.get("spib_ndim", 2))
        self.lagtime = int(plugin_config.get("lagtime", self.pcoord_len))

        # Encoder type ('Linear' or 'Nonlinear')
        if plugin_config.get("encoder_type", 'Linear') == 'Nonlinear':
            self.encoder_type = 'Nonlinear'
        else:
            self.encoder_type = 'Linear'

        self.neuron_num1 = int(plugin_config.get("neuron_num1", 16))
        self.neuron_num2 = int(plugin_config.get("neuron_num2", 16))
        self.batch_size = int(plugin_config.get("batch_size", 64))
        self.tolerance = float(plugin_config.get("tolerance", 0.001))
        self.patience = int(plugin_config.get("patience", 5))
        self.refinements = int(plugin_config.get("refinements", 15))
        self.learning_rate = float(plugin_config.get("learning_rate", 0.001))
        self.beta = float(plugin_config.get("beta", 0.001))

        self.initial_state_num = int(plugin_config.get("initial_state_num", 100))
        self.expected_occupied_bin_num = int(plugin_config.get("expected_occupied_bin_num", 100))

        if str2bool(plugin_config.get("enable_data_transform")):
            self.enable_data_transform = True
            westpa.rc.pstatus("Enabling data normalization")
        else:
            self.enable_data_transform = False

        if str2bool(plugin_config.get("enable_weights")):
            self.enable_weights = True
            westpa.rc.pstatus("Enabling weights")
        else:
            self.enable_weights = False

        # By default seed = 0
        # If it's changed, the path to saved files may also change
        self.seed = 0

        self.WE_folder = os.environ.get('WEST_SIM_ROOT')
        # Get the number of tasks and CPUs per task from environment variables
        num_tasks = int(os.getenv("SLURM_NTASKS", 1))
        cpus_per_task = int(os.getenv("SLURM_CPUS_PER_TASK", 1))
        # Calculate the total number of processes
        self.num_processes = num_tasks * cpus_per_task

        if self.WE_folder is None:
            log.error('Cannot find WEST_SIM_ROOT')
            raise AssertionError('Cannot find WEST_SIM_ROOT')

        self.west_file_path = self.WE_folder + '/west.h5'

        # create SPIB folder for saving results
        self.SPIB_folder = self.WE_folder + '/SPIB'
        if not os.path.exists(self.SPIB_folder):
            os.makedirs(self.SPIB_folder)

        # remove environment variable SPIB_model_folder if it has already been defined
        if os.environ.get("SPIB_MODEL_FOLDER"):
            log.info('SPIB_MODEL_FOLDER has already been defined')
            del os.environ['SPIB_MODEL_FOLDER']
            log.info('SPIB_MODEL_FOLDER has been deleted')

        self.bin_target_counts = westpa.rc.config.get(
            ["west", "system", "system_options", "bin_target_counts"]
        )

        # initialize bin mapper
        self.original_bin_mapper = self.system.bin_mapper
        log.info('initializing SPIB bin mapper')
        westpa.rc.pstatus("Initializing SPIB bin mapper")
        self.bin_mapper_path = self.SPIB_folder + '/bin_mapper.model'
        if os.path.isfile(self.bin_mapper_path):
            log.info('loading saved SPIB bin mapper')
            we_bin_mapper = torch.load(self.bin_mapper_path)
        else:
            log.info('creating default SPIB bin mapper')
            we_bin_mapper = SPIBBinMapper(self.original_bin_mapper, pcoord_ndim=self.pcoord_ndim, SPIB=None,
                                          spib_ndim=self.spib_ndim)

        self.system.bin_mapper = we_bin_mapper
        torch.save(we_bin_mapper, self.bin_mapper_path)

        # initialize bin target count
        self.system.bin_target_counts = self.get_bin_target_counts()

        # Big number is low priority -- this should run before anything else
        self.priority = plugin_config.get("priority", 0)

        sim_manager.register_callback(
            sim_manager.post_we, self.do_optimization, self.priority
        )

    def do_optimization(self):
        """
        Train SPIB model, and update WESTPA with new SPIB model and bin mapper. Then, continue
        the WE for more iterations.
        """

        # 0. Check current iteration
        n_iter = self.sim_manager.n_iter
        # just update bin occupancy if it doesn't meet the expected iteration
        if n_iter < self.initial_iter or (n_iter - self.initial_iter) % self.update_interval != 0:
            with self.data_manager.lock:
                iter_group = self.data_manager.get_iter_group(n_iter)

            pcoords = iter_group['pcoord'][()]
            self.system.bin_mapper.update_bin_population(pcoords[:, -1])
            self.system.bin_target_counts = self.get_bin_target_counts()
            # save model
            torch.save(self.system.bin_mapper, self.SPIB_folder + '/bin_mapper.model')
            return

        # create SPIB model subfolder
        SPIB_model_folder = self.SPIB_folder + '/' + str(n_iter).zfill(6)
        if not os.path.exists(SPIB_model_folder):
            os.makedirs(SPIB_model_folder)

        log.info('created SPIB model subfolder for iteration {:d}'.format(n_iter))

        # 1. Collect data
        if self.collect_last_n_iter <= self.update_interval:
            iter0 = max(1 + self.exclude_initial_n_iter, n_iter + 1 - self.collect_last_n_iter)
            dataset = self.collect_data(iter0, n_iter + 1)
        else:
            # get new dataset
            new_dataset = self.collect_data(max(1 + self.exclude_initial_n_iter, n_iter + 1 - self.update_interval), n_iter + 1)

            # get old dataset
            if n_iter >= self.collect_last_n_iter:
                iter0 = max(1 + self.exclude_initial_n_iter, n_iter + 1 - self.collect_last_n_iter)
                old_dataset = self.collect_old_data(iter0, n_iter + 1 - self.update_interval)
                dataset = np.concatenate([old_dataset, new_dataset], axis=0)
            elif n_iter > self.update_interval:
                old_dataset = self.collect_old_data(self.exclude_initial_n_iter + 1, n_iter + 1 - self.update_interval)
                dataset = np.concatenate([old_dataset, new_dataset], axis=0)
            else:
                dataset = new_dataset

            # save the new dataset
            np.save(SPIB_model_folder + '/new_dataset.npy', new_dataset)

        log.info('collected and saved trajectory data')

        # get the weights
        iter0 = max(self.exclude_initial_n_iter + 1, n_iter + 1 - self.collect_last_n_iter)
        weights = self.get_west_weights(iter0, n_iter + 1)

        traj_weights = np.zeros(dataset.shape[:2])
        for i in range(weights.shape[0]):
            traj_weights[i] = weights[i]
        np.save(SPIB_model_folder + '/traj_weights.npy', traj_weights)
        if self.enable_weights is False:
            traj_weights = None

        # collect the pcoord
        traj_pcoords = self.get_west_pcoord(iter0, n_iter + 1)

        # 2. Generate initial state labels
        # Use the expert-based CVs to generate initial state labels
        initial_labels = generate_initial_labels(traj_pcoords[:, :, :-self.spib_ndim], self.initial_state_num,
                                                 self.batch_size, SPIB_model_folder)

        # data shape
        data_shape = dataset.shape[-1:]
        output_dim = np.max(initial_labels) + 1

        # data normalization
        if self.enable_data_transform:
            data_transform = DataNormalize(mean=dataset.mean(axis=(0, 1)), std=dataset.std(axis=(0, 1)))
        else:
            data_transform = None

        # 3. Train SPIB model
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        ### Split the data
        # split data into train and test set
        indices = list(range(len(dataset)))
        split = int(np.floor(0.1 * len(dataset)))

        np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]

        log.info('training SPIB model')

        IB_path = SPIB_model_folder + '/SPIB'
        final_result_path = IB_path + '_result.dat'
        os.makedirs(os.path.dirname(final_result_path), exist_ok=True)
        if os.path.exists(final_result_path):
            print("Final Result", file=open(final_result_path, 'a'))  # append if already exists
        else:
            print("Final Result", file=open(final_result_path, 'w'))

        # prepare the dataset for spib training

        train_dataset, test_dataset = prepare_data(dataset, initial_labels, weight_list=traj_weights,
                                                   output_dim=output_dim, lagtime=self.lagtime,
                                                   train_indices=train_indices,
                                                   test_indices=test_indices, device=self.device)

        # Now a SPIB instance can be created. For the full range of possible arguments, please see the API docs.
        IB = SPIB(output_dim=output_dim, data_shape=data_shape, encoder_type=self.encoder_type, z_dim=self.spib_ndim,
                  lagtime=self.lagtime, beta=self.beta, learning_rate=self.learning_rate, device=self.device,
                  path=IB_path, UpdateLabel=True, neuron_num1=self.neuron_num1, neuron_num2=self.neuron_num2,
                  data_transform=data_transform)

        IB.to(self.device)

        IB.fit(train_dataset, test_dataset, batch_size=self.batch_size, tolerance=self.tolerance,
               patience=self.patience, refinements=self.refinements, index=self.seed)

        torch.save(IB, SPIB_model_folder + '/SPIB.model')
        torch.save(IB, self.SPIB_folder + '/SPIB.model')

        log.info('saved SPIB model and data files')

        # 4. Update bin mapper

        # load the saved state labels and state prediction
        traj_labels, traj_prediction, traj_z_latent, _ = IB.transform(dataset.reshape((-1, data_shape[0])),
                                                                      batch_size=self.batch_size, to_numpy=True)

        # update current pcoords
        with self.data_manager.lock:
            iter_group = self.data_manager.get_iter_group(n_iter)

        old_pcoords = iter_group['pcoord'][()].reshape(-1, self.pcoord_ndim)
        pcoords = np.concatenate([old_pcoords[:, :-self.spib_ndim], traj_z_latent[-old_pcoords.shape[0]:]], axis=-1)

        pcoords = pcoords.reshape([-1, self.pcoord_len, self.pcoord_ndim])

        log.info('updating bin mapper')
        westpa.rc.pstatus("Updating bin mapper")

        we_bin_mapper = SPIBBinMapper(self.original_bin_mapper, pcoord_ndim=self.pcoord_ndim, SPIB=IB,
                                      spib_ndim=self.spib_ndim, expected_spib_bin_num=self.expected_occupied_bin_num)

        # Binning along the learned SPIB space
        # update spib_bin_mapper
        we_bin_mapper.update_spib_bin_mapper(pcoords[:, -1])

        # update under-sampled bins
        we_bin_mapper.update_under_sampled_bins(traj_pcoords)

        # update bin population
        we_bin_mapper.update_bin_population(pcoords[:, -1])

        # Update bin mapper
        self.system.bin_mapper = we_bin_mapper

        # Save SPIB bin mapper
        torch.save(we_bin_mapper, SPIB_model_folder + '/bin_mapper.model')
        torch.save(we_bin_mapper, self.SPIB_folder + '/bin_mapper.model')

        # 5. Update bin target counts
        self.system.bin_target_counts = self.get_bin_target_counts()

        # save the new dataset as old_traj_data.npy for future usage
        np.save(self.SPIB_folder + '/old_traj_data.npy', dataset)
        log.info('collected and saved trajectory data')


    def get_traj_data(self, idx):
        folder = self.WE_folder + '/traj_segs/' + str(idx[0]).zfill(6) + '/'
        filename = folder + str(idx[1]).zfill(6) + '/traj_data.npy'
        return np.load(filename)

    def collect_data(self, iter0, iter1):
        # in the future, this step should be parallelized to enhance the efficiency
        # collecting data
        dataset = []

        start_time = time.time()
        with h5py.File(self.west_file_path, "r") as f:
            for i in range(iter0, iter1):
                n_seg = f['summary'][i - 1][0]
                for j in range(n_seg):
                    dataset.append(self.get_traj_data((i, j)))

        print("Collecting Data Time:--- %s seconds ---" % (time.time() - start_time))

        dataset = np.stack(dataset, axis=0)

        return dataset

    def collect_old_data(self, iter0, iter1):
        old_data_path = self.SPIB_folder + '/old_traj_data.npy'
        if os.path.isfile(old_data_path):
            log.info('loading old traj data')
            dataset = np.load(old_data_path)

            # only take the last iter1 - iter0
            dataset_len = 0

            with h5py.File(self.west_file_path, "r") as f:
                for i in range(iter0, iter1):
                    dataset_len += f['summary'][i - 1][0]

            dataset = dataset[-dataset_len:]

        else:
            dataset = self.collect_data(iter0, iter1)
            np.save(old_data_path, dataset)

        return dataset

    def get_west_weights(self, iter0, iter1):
        # extract the segment weights from west file
        weights = []
        with h5py.File(self.west_file_path, "r") as f:
            for i in list(f['iterations'])[(iter0 - 1):(iter1 - 1)]:
                seg_index = f['iterations'][i]['seg_index'][()]
                weights += [seg_index[i][0] for i in range(seg_index.shape[0])]

        weights = np.array(weights)

        return weights

    def get_west_pcoord(self, iter0, iter1):
        # extract the progress coordinate from west file
        pcoord = []
        with h5py.File(self.west_file_path, "r") as f:
            for i in list(f['iterations'])[(iter0 - 1):(iter1 - 1)]:
                pcoord += [f['iterations'][i]['pcoord'][()]]

        pcoord = np.concatenate(pcoord, 0)

        return pcoord

    def get_bin_target_counts(self):
        nbins = self.system.bin_mapper.nbins
        new_target_counts = np.empty((nbins,), dtype=int)
        new_target_counts[...] = self.bin_target_counts

        return new_target_counts

