# Warsaw University of Technology

import os
import configparser
import time
import numpy as np

from datasets.quantization import PolarQuantizer, CartesianQuantizer


class ModelParams:
    def __init__(self, model_params_path):
        config = configparser.ConfigParser()
        config.read(model_params_path)
        params = config['MODEL']

        self.model_params_path = model_params_path
        self.model = params.get('model')
        self.output_dim = params.getint('output_dim', 256)      # Size of the final descriptor

        #######################################################################
        # Model dependent
        #######################################################################

        self.coordinates = params.get('coordinates', 'polar')
        assert self.coordinates in ['polar', 'cartesian'], f'Unsupported coordinates: {self.coordinates}'

        if 'polar' in self.coordinates:
            # 3 quantization steps for polar coordinates: for sectors (in degrees), rings (in meters) and z coordinate (in meters)
            self.quantization_step = [float(e) for e in params['quantization_step'].split(',')]
            assert len(self.quantization_step) == 3, f'Expected 3 quantization steps: for sectors (degrees), rings (meters) and z coordinate (meters)'
            self.quantizer = PolarQuantizer(quant_step=self.quantization_step)
        elif 'cartesian' in self.coordinates:
            # Single quantization step for cartesian coordinates
            self.quantization_step = params.getfloat('quantization_step')
            self.quantizer = CartesianQuantizer(quant_step=self.quantization_step)
        else:
            raise NotImplementedError(f"Unsupported coordinates: {self.coordinates}")

        #self.remove_ground_plane = params.getboolean('remove_ground_plane', False)

        if 'MinkLoc' in self.model:
            # Size of the local features from backbone network (only for MinkNet based models)
            self.feature_size = params.getint('feature_size', 256)
            if 'planes' in params:
                self.planes = [int(e) for e in params['planes'].split(',')]
            else:
                self.planes = [32, 64, 64]

            if 'layers' in params:
                self.layers = [int(e) for e in params['layers'].split(',')]
            else:
                self.layers = [1, 1, 1]

            self.num_top_down = params.getint('num_top_down', 1)
            self.conv0_kernel_size = params.getint('conv0_kernel_size', 5)
            self.block = params.get('block', 'BasicBlock')
            self.pooling = params.get('pooling', 'GeM')

    def print(self):
        print('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e == 'quantization_step':
                s = param_dict[e]
                if self.coordinates == 'polar':
                    print(f'quantization_step - sector: {s[0]} [deg] / ring: {s[1]} [m] / z: {s[2]} [m]')
                else:
                    print(f'quantization_step: {s} [m]')
            else:
                print('{}: {}'.format(e, param_dict[e]))

        print('')


def get_datetime():
    return time.strftime("%Y%m%d_%H%M")


class TrainingParams:
    """
    Parameters for model training
    """
    def __init__(self, params_path, model_params_path):
        """
        Configuration files
        :param path: Training configuration file
        :param model_params: Model-specific configuration file
        """

        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(model_params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path

        config = configparser.ConfigParser()

        config.read(self.params_path)
        params = config['DEFAULT']
        self.dataset = params.get('dataset', 'mulran').lower()
        self.dataset_folder = params.get('dataset_folder')
        # Seconday dataset for global descriptor training
        self.secondary_dataset = params.get('secondary_dataset', None)
        if self.secondary_dataset is not None:
            self.secondary_dataset = self.secondary_dataset.lower()
        self.secondary_dataset_folder = params.get('secondary_dataset_folder', None)

        # Maximum random rotation and translation applied when generating pairs for local descriptor
        self.rot_max = params.getfloat('rot_max', np.pi)
        self.trans_max = params.getfloat('rot_max', 5.)

        params = config['TRAIN']

        self.save_freq = params.getint('save_freq', 20)          # Model saving frequency (in epochs)
        self.num_workers = params.getint('num_workers', 4)
        # Initial batch size for global descriptors (for both main and secondary dataset)
        self.batch_size = params.getint('batch_size', 64)
        # Batch size for local descriptors
        self.local_batch_size = params.getint('local_batch_size', 2)

        # Set batch_expansion_th to turn on dynamic batch sizing
        # When number of non-zero triplets falls below batch_expansion_th, expand batch size
        self.batch_expansion_th = params.getfloat('batch_expansion_th', None)
        if self.batch_expansion_th is not None:
            assert 0. < self.batch_expansion_th < 1., 'batch_expansion_th must be between 0 and 1'
            self.batch_size_limit = params.getint('batch_size_limit', 256)
            # Batch size expansion rate
            self.batch_expansion_rate = params.getfloat('batch_expansion_rate', 1.5)
            assert self.batch_expansion_rate > 1., 'batch_expansion_rate must be greater than 1'
        else:
            self.batch_size_limit = self.batch_size
            self.batch_expansion_rate = None

        if 'secondary_batch_size_limit' in params:
            self.secondary_batch_size_limit = params.getint('secondary_batch_size_limit')
        else:
            self.secondary_batch_size_limit = self.batch_size_limit

        self.loss_gammas = params.get('l_gammas', None)
        if self.loss_gammas is not None:
            self.loss_gammas = [float(e) for e in self.loss_gammas.split(',')]
        self.lr = params.getfloat('lr', 1e-3)

        self.scheduler = params.get('scheduler', 'MultiStepLR')
        if self.scheduler is not None:
            if self.scheduler == 'CosineAnnealingLR':
                self.min_lr = params.getfloat('min_lr')
            elif self.scheduler == 'MultiStepLR':
                scheduler_milestones = params.get('scheduler_milestones')
                self.scheduler_milestones = [int(e) for e in scheduler_milestones.split(',')]
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.scheduler))

        self.epochs = params.getint('epochs', 20)
        self.weight_decay = params.getfloat('weight_decay', None)
        self.loss = params.get('loss')

        if 'Contrastive' in self.loss:
            self.pos_margin = params.getfloat('pos_margin', 0.2)
            self.neg_margin = params.getfloat('neg_margin', 0.65)
        elif 'Triplet' in self.loss:
            self.margin = params.getfloat('margin', 0.4)    # Margin used in loss function
        else:
            raise 'Unsupported loss function: {}'.format(self.loss)

        self.aug_mode = params.getint('aug_mode', 1)    # Augmentation mode (1 is default)

        self.train_file = params.get('train_file')
        self.val_file = params.get('val_file', None)
        self.secondary_train_file = params.get('secondary_train_file', None)
        self.test_file = params.get('test_file', None)

        # Read model parameters
        self.model_params = ModelParams(self.model_params_path)
        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e != 'model_params':
                print('{}: {}'.format(e, param_dict[e]))

        self.model_params.print()
        print('')

