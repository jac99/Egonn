# Functions and classes operating on a raw Mulran dataset

import numpy as np
import os
from typing import List
from torch.utils.data import Dataset, ConcatDataset
from sklearn.neighbors import KDTree
import torch

from datasets.mulran.utils import read_lidar_poses, in_test_split, in_train_split
from misc.point_clouds import PointCloudLoader


class MulranPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level.
        self.ground_plane_level = -0.9

    def read_pc(self, file_pathname: str) -> torch.Tensor:
        # Reads the point cloud without pre-processing
        # Returns Nx3 tensor
        pc = np.fromfile(file_pathname, dtype=np.float32)
        # PC in Mulran is of size [num_points, 4] -> x,y,z,reflectance
        pc = np.reshape(pc, (-1, 4))[:, :3]
        return pc


class MulranSequence(Dataset):
    """
    Dataset returns a point cloud from a train or test split from one sequence from a raw Mulran dataset
    """
    def __init__(self, dataset_root: str, sequence_name: str, split: str, min_displacement: float = 0.2):
        assert os.path.exists(dataset_root), f'Cannot access dataset root: {dataset_root}'
        assert split in ['train', 'test', 'all']

        self.dataset_root = dataset_root
        self.sequence_name = sequence_name
        sequence_path = os.path.join(self.dataset_root, self.sequence_name)
        assert os.path.exists(sequence_path), f'Cannot access sequence: {sequence_path}'
        self.split = split
        self.min_displacement = min_displacement
        # Maximum discrepancy between timestamps of LiDAR scan and global pose in seconds
        self.pose_time_tolerance = 1.

        self.pose_file = os.path.join(sequence_path, 'global_pose.csv')
        assert os.path.exists(self.pose_file), f'Cannot access global pose file: {self.pose_file}'

        self.rel_lidar_path = os.path.join(self.sequence_name, 'Ouster')
        lidar_path = os.path.join(self.dataset_root, self.rel_lidar_path)
        assert os.path.exists(lidar_path), f'Cannot access lidar scans: {lidar_path}'
        self.pc_loader = MulranPointCloudLoader()

        timestamps, poses = read_lidar_poses(self.pose_file, lidar_path, self.pose_time_tolerance)
        self.timestamps, self.poses = self.filter(timestamps, poses)
        self.rel_scan_filepath = [os.path.join(self.rel_lidar_path, str(e) + '.bin') for e in self.timestamps]

        assert len(self.timestamps) == len(self.poses)
        assert len(self.timestamps) == len(self.rel_scan_filepath)
        print(f'{len(self.timestamps)} scans in {sequence_name}-{split}')

    def __len__(self):
        return len(self.rel_scan_filepath)

    def __getitem__(self, ndx):
        reading_filepath = os.path.join(self.dataset_root, self.rel_scan_filepath[ndx])
        reading = self.pc_loader(reading_filepath)
        return {'pc': reading, 'pose': self.poses[ndx], 'ts': self.timestamps[ndx],
                'position': self.poses[ndx][:2, 3]}

    def filter(self, ts: np.ndarray, poses: np.ndarray):
        # Filter out scans - retain only scans within a given split with minimum displacement
        positions = poses[:, :2, 3]

        # Retain elements in the given split
        # Only sejong sequence has train/test split
        if self.split != 'all' and self.sequence_name.lower()[:6] == 'sejong':
            if self.split == 'train':
                mask = in_train_split(positions)
            elif self.split == 'test':
                mask = in_test_split(positions)

            ts = ts[mask]
            poses = poses[mask]
            positions = positions[mask]
            #print(f'Split: {self.split}   Mask len: {len(mask)}   Mask True: {np.sum(mask)}')

        # Filter out scans - retain only scans within a given split
        prev_position = None
        mask = []
        for ndx, position in enumerate(positions):
            if prev_position is None:
                mask.append(ndx)
            else:
                displacement = np.linalg.norm(prev_position - position)
                if displacement > self.min_displacement:
                    mask.append(ndx)
                    prev_position = position

        ts = ts[mask]
        poses = poses[mask]
        return ts, poses


class MulranSequences(Dataset):
    """
    Multiple Mulran sequences indexed as a single dataset. Each element is identified by a unique global index.
    """
    def __init__(self, dataset_root: str, sequence_names: List[str], split: str, min_displacement: float = 0.2):
        assert len(sequence_names) > 0
        assert os.path.exists(dataset_root), f'Cannot access dataset root: {dataset_root}'
        assert split in ['train', 'test', 'all']

        self.dataset_root = dataset_root
        self.sequence_names = sequence_names
        self.split = split
        self.min_displacement = min_displacement

        sequences = []
        for seq_name in self.sequence_names:
            ds = MulranSequence(self.dataset_root, seq_name, split=split, min_displacement=min_displacement)
            sequences.append(ds)

        self.dataset = ConcatDataset(sequences)

        # Concatenate positions from all sequences
        self.poses = np.zeros((len(self.dataset), 4, 4), dtype=np.float64)
        self.timestamps = np.zeros((len(self.dataset),), dtype=np.int64)
        self.rel_scan_filepath = []

        for cum_size, ds in zip(self.dataset.cumulative_sizes, sequences):
            # Consolidated lidar positions, timestamps and relative filepaths
            self.poses[cum_size - len(ds): cum_size, :] = ds.poses
            self.timestamps[cum_size - len(ds): cum_size] = ds.timestamps
            self.rel_scan_filepath.extend(ds.rel_scan_filepath)

        assert len(self.timestamps) == len(self.poses)
        assert len(self.timestamps) == len(self.rel_scan_filepath)

        # Build a kdtree based on X, Y position
        self.kdtree = KDTree(self.get_xy())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ndx):
        return self.dataset[ndx]

    def get_xy(self):
        # Get X, Y position from (4, 4) pose
        return self.poses[:, :2, 3]

    def find_neighbours_ndx(self, position, radius):
        # Returns indices of neighbourhood point clouds for a given position
        assert position.ndim == 1
        assert position.shape[0] == 2
        # Reshape into (1, 2) axis
        position = position.reshape(1, -1)
        neighbours = self.kdtree.query_radius(position, radius)[0]
        return neighbours.astype(np.int32)


if __name__ == '__main__':
    dataset_root = '/media/sf_Datasets/MulRan'
    sequence_names = ['Sejong01']

    db = MulranSequences(dataset_root, sequence_names, split='train')
    print(f'Number of scans in the sequence: {len(db)}')
    e = db[0]

    res = db.find_neighbours_ndx(e['position'], radius=50)
    print('.')



