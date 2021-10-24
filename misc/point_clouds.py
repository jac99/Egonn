import copy
import os

import numpy as np
import open3d as o3d


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def draw_pc(pc):
    pc = copy.deepcopy(pc)
    pc.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([pc],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def icp(anchor_pc, positive_pc, transform: np.ndarray = None, point2plane: bool = False,
        inlier_dist_threshold: float = 1.2, max_iteration: int = 200):
    # transform: initial alignment transform
    if transform is not None:
        transform = transform.astype(float)

    voxel_size = 0.1
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(anchor_pc)
    pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(positive_pc)
    pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)

    if point2plane:
        pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        transform_estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        transform_estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    if transform is not None:
        reg_p2p = o3d.pipelines.registration.registration_icp(pcd1, pcd2, inlier_dist_threshold, transform,
                                                              estimation_method=transform_estimation,
                                                              criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    else:
        reg_p2p = o3d.pipelines.registration.registration_icp(pcd1, pcd2, inlier_dist_threshold,
                                                              estimation_method=transform_estimation,
                                                              criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

    return reg_p2p.transformation, reg_p2p.fitness, reg_p2p.inlier_rmse


def make_open3d_feature(data, dim, npts):
    feature = o3d.pipelines.registration.Feature()
    feature.resize(dim, npts)
    feature.data = data.cpu().numpy().astype('d').transpose()
    return feature


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


class PointCloudLoader:
    # Generic point cloud loader class
    def __init__(self):
        # remove_zero_points: remove points with all zero coordinates
        # remove_ground_plane: remove points on ground plane level and below
        # ground_plane_level: ground plane level
        self.remove_zero_points = True
        self.remove_ground_plane = True
        self.ground_plane_level = None
        self.set_properties()

    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level. Must be defined in inherited classes.
        raise NotImplementedError('set_properties must be defined in inherited classes')

    def __call__(self, file_pathname):
        # Reads the point cloud from a disk and preprocess (optional removal of zero points and points on the ground
        # plane and below
        # file_pathname: relative file path
        assert os.path.exists(file_pathname), f"Cannot open point cloud: {file_pathname}"
        pc = self.read_pc(file_pathname)
        assert pc.shape[1] == 3

        if self.remove_zero_points:
            mask = np.all(np.isclose(pc, 0), axis=1)
            pc = pc[~mask]

        if self.remove_ground_plane:
            mask = pc[:, 2] > self.ground_plane_level
            pc = pc[mask]

        return pc

    def read_pc(self, file_pathname):
        # Reads the point cloud without pre-processing
        raise NotImplementedError("read_pc must be overloaded in an inheriting class")
