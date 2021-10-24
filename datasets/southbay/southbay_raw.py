# Classes and functions to read Apollo SouthBay dataset

import os
import numpy as np
import csv
from typing import List

import third_party.pypcd as pypcd
import misc.poses as poses
from misc.point_clouds import PointCloudLoader


class GroundTruthPoses:
    def __init__(self, pose_filepath):
        assert os.path.isfile(pose_filepath), f'Cannot access pose file: {pose_filepath}'
        self.pose_filepath = pose_filepath
        self.pose_ndx = {}
        self.read_poses()

    def read_poses(self):
        with open(self.pose_filepath) as h:
            csv_reader = csv.reader(h, delimiter=' ')
            for ndx, row in enumerate(csv_reader):
                assert len(row) == 9, f'Incorrect format of row {ndx}: {row}'
                ndx = int(row[0])
                ts = float(row[1])
                x = float(row[2])
                y = float(row[3])
                z = float(row[4])
                qx = float(row[5])
                qy = float(row[6])
                qz = float(row[7])
                qr = float(row[8])
                se3 = np.eye(4, dtype=np.float64)
                # Expects quaternions in w, x, y, z format
                se3[0:3, 0:3] = poses.q2r((qr, qx, qy, qz))
                se3[0:3, 3] = np.array([x, y, z])
                self.pose_ndx[ndx] = (se3, ts)      # (pose, timestamp)


class PointCloud:
    id: int = 0         # global PointCloud id (unique for each cloud)

    def __init__(self, rel_scan_filepath: str, pose: np.ndarray, timestamp: float):
        self.rel_scan_filepath = rel_scan_filepath
        self.pose = pose
        self.timestamp = timestamp
        filename = os.path.split(rel_scan_filepath)[1]
        # Relative point cloud ids start from 1 in each subfolder/traversal
        self.rel_id = int(os.path.splitext(filename)[0])
        self.id = PointCloud.id
        PointCloud.id += 1


class SouthBayDataset:
    def __init__(self, dataset_root):
        assert os.path.isdir(dataset_root), f'Cannot access directory: {dataset_root}'
        self.dataset_root = dataset_root

        self.splits = ['MapData', 'TestData', 'TrainData']
        self.pcd_extension = '.pcd'     # Point cloud extension

        # location_ndx[split][location] = [... list of global ids in this location ...]
        self.location_ndx = {}

        # pc_ndc[global_id] = PointCloud
        self.global_ndx = {}

        for split in self.splits:
            self.location_ndx[split] = {}
            self.index_split(split)

    def index_split(self, split):
        path = os.path.join(self.dataset_root, split)
        assert os.path.isdir(path), f"Missing split: {split}"

        # Index locations
        locations = os.listdir(path)
        locations = [f for f in locations if os.path.isdir(os.path.join(path, f))]
        locations.sort()
        for loc in locations:
            # Locations may contain multiple subfolders hierachy
            # All point clouds in all subfolders are stored as one list
            rel_working_path = os.path.join(split, loc)
            self.location_ndx[split][loc] = []
            self.index_location(split, loc, rel_working_path)

    def index_location(self, split, loc, rel_working_path):
        working_path = os.path.join(self.dataset_root, rel_working_path)
        subfolders = os.listdir(working_path)
        if 'pcds' in subfolders and 'poses' in subfolders:
            # Process point clouds and poses
            rel_pcds_path = os.path.join(rel_working_path, 'pcds')
            poses_path = os.path.join(working_path, 'poses')
            poses_filepath = os.path.join(poses_path, 'gt_poses.txt')
            assert os.path.isfile(poses_filepath), f'Missing poses file: {poses_filepath}'
            tp = GroundTruthPoses(poses_filepath)
            for e in tp.pose_ndx:
                se3, ts = tp.pose_ndx[e]
                rel_pcd_filepath = os.path.join(rel_pcds_path, str(e) + self.pcd_extension)
                pcd_filepath = os.path.join(self.dataset_root, rel_pcd_filepath)
                if not os.path.exists(pcd_filepath):
                    print(f'Missing pcd file: {pcd_filepath}')
                pc = PointCloud(rel_pcd_filepath, se3, ts)
                self.global_ndx[pc.id] = pc
                self.location_ndx[split][loc].append(pc.id)
        elif 'pcds' in subfolders or 'poses' in subfolders:
            assert False, 'Something wrong. Either pcds or poses folder is missing'

        # Recursively process other subfolders - check if they contain point data
        rel_subfolders = [os.path.join(rel_working_path, p) for p in subfolders]
        rel_subfolders = [p for p in rel_subfolders if os.path.isdir(os.path.join(self.dataset_root, p))]

        for sub in rel_subfolders:
            self.index_location(split, loc, sub)

    def print_info(self):
        print(f'Dataset root: {self.dataset_root}')
        print(f"Splits: {self.splits}")
        for split in self.location_ndx:
            locations = self.location_ndx[split].keys()
            print(f"Locations in {split}: {locations}")
            for loc in locations:
                pc_list = self.location_ndx[split][loc]
                print(f"{len(pc_list)} point clouds in location {split} - {loc}")
            print("")
        print(f'Last point cloud id: {PointCloud.id - 1}')

    def get_poses(self, split, location=None):
        # Get ids and poses of all point clouds from the given split and optionally within a location
        if location is None:
            locations = list(self.location_ndx[split])
        else:
            locations = [location]

        # Count point clouds
        count_pc = 0
        for loc in locations:
            count_pc += len(self.location_ndx[split][loc])

        # Point cloud global ids
        pc_ids = np.zeros(count_pc, dtype=np.int64)
        # Poses
        pc_poses = np.zeros((count_pc, 4, 4), dtype=np.float64)

        # Fill ids and pose tables
        n = 0
        for loc in locations:
            for pc_id in self.location_ndx[split][loc]:
                pc = self.global_ndx[pc_id]
                pc_ids[n] = pc_id
                pc_poses[n] = pc.pose
                n += 1

        return pc_ids, pc_poses

    def get_poses2(self, splits: List[str]):
        # Get ids and poses of all point clouds from the given splits

        locations = list(self.location_ndx[splits[0]])
        print(f"Locations: {locations}")

        # Count point clouds
        count_pc = 0
        for split in splits:
            for loc in locations:
                count_pc += len(self.location_ndx[split][loc])

        # Point cloud global ids
        pc_ids = np.zeros(count_pc, dtype=np.int32)
        # Poses
        pc_poses = np.zeros((count_pc, 4, 4), dtype=np.float64)

        # Fill ids and pose tables
        n = 0
        for split in splits:
            for loc in locations:
                for pc_id in self.location_ndx[split][loc]:
                    pc = self.global_ndx[pc_id]
                    pc_ids[n] = pc_id
                    pc_poses[n] = pc.pose
                    n += 1

        return pc_ids, pc_poses


class SouthbayPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Set point cloud propertiers, such as ground_plane_level. Must be defined in inherited classes.
        self.ground_plane_level = -1.6

    def read_pc(self, file_pathname):
        pc = pypcd.PointCloud.from_path(file_pathname)
        # pc.pc_data has the data as a structured array
        # pc.fields, pc.count, etc have the metadata
        pc = np.stack([pc.pc_data['x'], pc.pc_data['y'], pc.pc_data['z']], axis=1)
        # Replace naans with all-zero coords
        nan_mask = np.isnan(pc).any(axis=1)
        pc[nan_mask] = np.array([0., 0., 0.], dtype=np.float)
        return pc
