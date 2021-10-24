# Code based on ScanContext implementation: https://github.com/irapkaist/scancontext/blob/master/python/make_sc_example.py
# Partially vectorized implementation by Jacek Komorowski

import numpy as np
import numpy_indexed as npi
from sklearn.neighbors import KDTree


def pt2rs(points, gap_ring, gap_sector):
    # np.arctan2 produces values in -pi..pi range
    theta = np.arctan2(points[:, 1], points[:, 0]) + np.pi
    eps = 1e-6

    theta = np.clip(theta, a_min=0., a_max=2*np.pi-eps)
    faraway = np.linalg.norm(points[:, 0:2], axis=1)

    idx_ring = (faraway // gap_ring).astype(int)
    idx_sector = (theta // gap_sector).astype(int)

    return idx_ring, idx_sector


class ScanContext:
    def __init__(self, num_sector=60, num_ring=20, max_length=80, lidar_height=2.0):
        # lidar_height: app. lidar height above the ground, for Kitti is set to 2.0
        self.lidar_height = lidar_height
        self.num_sector = num_sector
        self.num_ring = num_ring
        self.max_length = max_length
        self.gap_ring = self.max_length / self.num_ring
        self.gap_sector = 2. * np.pi / self.num_sector

    def __call__(self, x):
        idx_ring, idx_sector = pt2rs(x, self.gap_ring, self.gap_sector)
        height = x[:, 2] + self.lidar_height

        # Filter out points that are self.max_length or further away
        mask = idx_ring < self.num_ring
        idx_ring = idx_ring[mask]
        idx_sector = idx_sector[mask]
        height = height[mask]

        assert idx_ring.shape == idx_sector.shape
        assert idx_ring.shape == height.shape

        # Convert idx_ring and idx_sector to a linear index
        idx_linear = idx_ring * self.num_sector + idx_sector
        idx, max_height = npi.group_by(idx_linear).max(height)

        sc = np.zeros([self.num_ring, self.num_sector])
        # As per original ScanContext implementation, the minimum height is always non-negative
        # (they take max from 0 and height values)
        sc[idx // self.num_sector, idx % self.num_sector] = np.clip(max_height, a_min=0., a_max=None)

        return sc


def distance_sc(sc1, sc2):
    # Distance between 2 scan context descriptors
    num_sectors = sc1.shape[1]

    # Repeate to move 1 columns
    _one_step = 1  # const
    sim_for_each_cols = np.zeros(num_sectors)
    for i in range(num_sectors):
        # Shift
        sc1 = np.roll(sc1, _one_step, axis=1)  # columne shift

        # Compare
        sc1_norm = np.linalg.norm(sc1, axis=0)
        sc2_norm = np.linalg.norm(sc2, axis=0)
        mask = ~np.logical_or(np.isclose(sc1_norm, 0.), np.isclose(sc2_norm, 0.))

        # Compute cosine similarity between columns of sc1 and sc2
        cossim = np.sum(np.multiply(sc1[:, mask], sc2[:, mask]), axis=0) / (sc1_norm[mask] * sc2_norm[mask])

        sim_for_each_cols[i] = np.sum(cossim) / np.sum(mask)

    yaw_diff = (np.argmax(sim_for_each_cols) + 1) % sc1.shape[1]  # because python starts with 0
    sim = np.max(sim_for_each_cols)
    dist = 1. - sim

    return dist, yaw_diff


def sc2rk(sc):
    # Scan context to ring key
    return np.mean(sc, axis=1)


class ScanContextManager:
    def __init__(self, num_sector=60, num_ring=20, max_length=80, lidar_height=2.0, max_capacity=100000):
        # max_capacity: maximum number of nodes
        self.num_sector = num_sector
        self.num_ring = num_ring
        self.max_length = max_length
        self.lidar_height = lidar_height
        self.max_capacity = max_capacity

        self.sc = ScanContext(self.num_sector, self.num_ring, self.max_length, self.lidar_height)
        self.scancontexts = np.zeros((self.max_capacity, self.num_ring, self.num_sector))
        self.ringkeys = np.zeros((self.max_capacity, self.num_ring))
        self.curr_node_idx = 0
        self.knn_tree_idx = -1      # Number of nodes for which KDTree tree was computed (recalculate KDTree only after new nodes are added)
        self.ringkey_tree = None

    def add_node(self, pc):
        # Compute and store the point cloud descriptor
        # pc: (N, 3) point cloud
        assert pc.ndim == 2
        assert pc.shape[1] == 3

        # Compute
        sc = self.sc(pc)        # (num_ring, num_sector) array
        rk = sc2rk(sc)          # (num_ring,) array

        self.scancontexts[self.curr_node_idx] = sc
        self.ringkeys[self.curr_node_idx] = rk
        self.curr_node_idx += 1
        assert self.curr_node_idx < self.max_capacity, f'Maximum ScanContextManager capacity exceeded: {self.max_capacity}'

    def query(self, query_pc, k=1, reranking=True):
        assert self.curr_node_idx > 0, 'Empty database'

        if self.curr_node_idx != self.knn_tree_idx:
            # recalculate KDTree
            self.ringkey_tree = KDTree(self.ringkeys[:self.curr_node_idx-1])
            self.knn_tree_idx = self.curr_node_idx

        # find candidates
        query_sc = self.sc(query_pc)        # (num_ring, num_sector) array
        query_rk = sc2rk(query_sc)          # (num_ring,) array
        # KDTree query expects (n_samples, sample_dimension) array
        _, nn_ndx = self.ringkey_tree.query(query_rk.reshape(1, -1), k=k)
        # nncandidates_idx is (n_samples=1, sample_dimension) array
        nn_ndx = nn_ndx[0]

        # step 2
        sc_dist = np.zeros((k,))
        sc_yaw_diff = np.zeros((k,))

        if not reranking:
            return nn_ndx, None, None

        # Reranking using the full ScanContext descriptor
        for i in range(k):
            candidate_ndx = nn_ndx[i]
            candidate_sc = self.scancontexts[candidate_ndx]
            sc_dist[i], sc_yaw_diff[i] = distance_sc(candidate_sc, query_sc)

        reranking_order = np.argsort(sc_dist)
        nn_ndx = nn_ndx[reranking_order]
        sc_yaw_diff = sc_yaw_diff[reranking_order]
        sc_dist = sc_dist[reranking_order]

        return nn_ndx, sc_dist, sc_yaw_diff
