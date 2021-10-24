# Warsaw University of Technology
# Dataset wrapper for Mulran lidar scans dataset

import os
import random
import numpy as np
import torch

from datasets.base_datasets import TrainingDataset
from datasets.quantization import Quantizer
from misc.poses import apply_transform
from datasets.base_datasets import TrainingDataset

DEBUG = False


class MulranTraining6DOFDataset(TrainingDataset):
    """
    Dataset wrapper for Mulran dataset for 6dof estimation.
    """
    def __init__(self, dataset_path: str, query_filename: str, quantizer: Quantizer,
                 rot_max: float = 0., trans_max: float = 0., **vargs):
        dataset_type = 'mulran'
        super().__init__(dataset_path, dataset_type, query_filename, **vargs)
        self.quantizer = quantizer
        self.rot_max = rot_max
        self.trans_max = trans_max

    def __getitem__(self, ndx):
        # pose is a global coordinate system pose 3x4 R|T matrix
        query_pc, _ = super().__getitem__(ndx)

        # get random positive
        positives = self.get_positives(ndx)
        positive_idx = np.random.choice(positives, 1)[0]
        positive_pc, _ = super().__getitem__(positive_idx)

        # get relative pose taking two global poses
        transform = self.queries[ndx].positives_poses[positive_idx]

        # Apply random transform to the positive point cloud
        rotation_angle = np.random.uniform(low=-self.rot_max, high=self.rot_max)
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        m = torch.eye(4, dtype=torch.float)
        #m[:3, :3] = np.array([[cosval, sinval, 0.], [-sinval, cosval, 0.], [0., 0., 1.]])
        m[:3, :3] = torch.tensor([[cosval, sinval, 0.], [-sinval, cosval, 0.], [0., 0., 1.]], dtype=m.dtype)
        m[:2, 3] = torch.rand((1, 2)) * 2. * self.trans_max - self.trans_max
        positive_pc = apply_transform(positive_pc, m)
        transform = m @ transform

        # Find indices of unique quantized coordinates and filter out points to leave max 1 point per voxel
        coords1, idx1 = self.quantizer(query_pc)
        coords2, idx2 = self.quantizer(positive_pc)
        pc1_cop = query_pc[idx1, :]
        pc2_trans_cop = positive_pc[idx2, :]

        return pc1_cop, pc2_trans_cop, transform
