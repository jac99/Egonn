# Functions and classes operating on a raw Kitti dataset

import numpy as np
import os
import torch
from torch.utils.data import Dataset

from misc.point_clouds import PointCloudLoader


class KittiPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level.
        self.ground_plane_level = -1.5

    def read_pc(self, file_pathname: str) -> torch.Tensor:
        # Reads the point cloud without pre-processing
        # Returns Nx3 tensor
        pc = np.fromfile(file_pathname, dtype=np.float32)
        # PC in Mulran is of size [num_points, 4] -> x,y,z,reflectance
        pc = np.reshape(pc, (-1, 4))[:, :3]
        return pc


class KittiSequence(Dataset):
    """
    Point cloud from a sequence from a raw Mulran dataset
    """
    def __init__(self, dataset_root: str, sequence_name: str, pose_time_tolerance: float = 1.,
                 remove_zero_points: bool = True):
        # pose_time_tolerance: (in seconds) skip point clouds without corresponding pose information (based on
        #                      timestamps difference)
        # remove_zero_points: remove (0,0,0) points

        assert os.path.exists(dataset_root), f'Cannot access dataset root: {dataset_root}'
        self.dataset_root = dataset_root
        self.sequence_name = sequence_name
        # self.sequence_path = os.path.join(self.dataset_root, 'sequences')
        # assert os.path.exists(self.sequence_path), f'Cannot access sequence: {self.sequence_path}'
        self.rel_lidar_path = os.path.join('sequences', self.sequence_name, 'velodyne')
        # lidar_path = os.path.join(self.sequence_path, self.rel_lidar_path)
        # assert os.path.exists(lidar_path), f'Cannot access lidar scans: {lidar_path}'
        self.pose_file = os.path.join(self.dataset_root, 'poses', self.sequence_name + '.txt')
        assert os.path.exists(self.pose_file), f'Cannot access sequence pose file: {self.pose_file}'
        self.times_file = os.path.join(self.dataset_root, 'sequences', self.sequence_name, 'times.txt')
        assert os.path.exists(self.pose_file), f'Cannot access sequence times file: {self.times_file}'
        # Maximum discrepancy between timestamps of LiDAR scan and global pose in seconds
        self.pose_time_tolerance = pose_time_tolerance
        self.remove_zero_points = remove_zero_points

        self.rel_lidar_timestamps, self.lidar_poses, filenames = self._read_lidar_poses()
        self.rel_scan_filepath = [os.path.join(self.rel_lidar_path, '%06d%s' % (e, '.bin')) for e in filenames]
        print(self.rel_scan_filepath)

    def __len__(self):
        return len(self.rel_lidar_timestamps)

    def __getitem__(self, ndx):
        scan_filepath = os.path.join(self.dataset_root, self.rel_scan_filepath[ndx])
        pc = load_pc(scan_filepath)
        if self.remove_zero_points:
            mask = np.all(np.isclose(pc, 0), axis=1)
            pc = pc[~mask]
        return {'pc': pc, 'pose': self.lidar_poses[ndx], 'ts': self.rel_lidar_timestamps[ndx]}

    def _read_lidar_poses(self):
        fnames = os.listdir(os.path.join(self.dataset_root, self.rel_lidar_path))
        assert len(fnames) > 0, f"Make sure that the path {self.rel_lidar_path}"
        filenames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])
        with open(self.pose_file, "r") as h:
            txt_poses = h.readlines()

        n = len(txt_poses)
        poses = np.zeros((n, 4, 4), dtype=np.float64)  # 4x4 pose matrix

        for ndx, pose in enumerate(txt_poses):
            # Split by comma and remove whitespaces
            temp = [e.strip() for e in pose.split(' ')]
            assert len(temp) == 12, f'Invalid line in global poses file: {temp}'
            # poses in kitti ar ein cam0 reference
            print(temp)
            poses[ndx] = np.array([[float(temp[0]), float(temp[1]), float(temp[2]), float(temp[3])],
                                   [float(temp[4]), float(temp[5]), float(temp[6]), float(temp[7])],
                                   [float(temp[8]), float(temp[9]), float(temp[10]), float(temp[11])],
                                   [0., 0., 0., 1.]])
        rel_ts = np.genfromtxt(self.times_file)

        return rel_ts, poses, filenames


def load_pc(filepath):
    # Load point cloud, does not apply any transform
    # Returns Nx3 matrix
    pc = np.fromfile(filepath, dtype=np.float32)
    # PC in Kitti is of size [num_points, 4] -> x,y,z,reflectance
    pc = np.reshape(pc, (-1, 4))[:, :3]
    return pc
