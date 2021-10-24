import numpy as np
import open3d as o3d
import os

from datasets.base_datasets import EvaluationTuple, EvaluationSet, get_pointcloud_loader
from misc.poses import apply_transform
from misc.point_clouds import draw_registration_result
from misc.poses import relative_pose


def check_alignment(pc1, pc2, inlier_dist_threshold=1., voxel_size=0.3):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2)
    pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)

    trans_init = np.eye(4, dtype=float)
    evaluation = o3d.pipelines.registration.evaluate_registration(pcd1, pcd2, inlier_dist_threshold, trans_init)
    print(evaluation)


if __name__ == '__main__':
    data_root = '/media/sf_Datasets/ApolloSouthBay'
    eval_set = 'southbay_BaylandsToSeafood_eval.pickle'
    eval_set_filepath = os.path.join(data_root, eval_set)

    voxel_size = 0.1
    inlier_dist_threshold = 1.2      # Inlier dist threshold for ICP registration

    eval_set = EvaluationSet()
    eval_set.load(eval_set_filepath)

    map_positions = eval_set.get_map_positions()
    query_positions = eval_set.get_query_positions()

    query_ndx = 6
    query_pos = query_positions[query_ndx]
    query_pose = eval_set.query_set[query_ndx].pose

    map_ndx = 16
    map_pos = map_positions[map_ndx]
    map_pose = eval_set.map_set[map_ndx].pose

    query_path = eval_set.query_set[query_ndx].rel_scan_filepath
    query_path = os.path.join(data_root, query_path)

    map_path = eval_set.map_set[map_ndx].rel_scan_filepath
    map_path = os.path.join(data_root, map_path)

    pc_loader = get_pointcloud_loader('southbay')
    pc1 = pc_loader(query_path)
    pc2 = pc_loader(map_path)

    pc1 = pc1[pc1[:,2] > -1.6]
    pc2 = pc2[pc2[:,2] > -1.6]

    delta = query_pos - map_positions[map_ndx]  # (k, 2) array
    euclid_dist = np.linalg.norm(delta)  # (k,) array

    T_gt = relative_pose(query_pose, map_pose)

    pc1_transformed = apply_transform(pc1, T_gt)        # pc1 transformed into second point cloud reference frame
    check_alignment(pc1_transformed, pc2, inlier_dist_threshold=inlier_dist_threshold)

    dist = np.linalg.norm(T_gt[:2, 3])
    print(f'Distance between clouds: {dist:0.2f} [m]')
    # Remove ground plane

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2)
    pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)

    print('Raw point clouds')

    trans_init = np.asarray([[1., 0., 0., 0.],
                             [0., 1., 0., 0.],
                             [0., 0., 1., 0.],
                             [.0, .0, .0, 1.]], dtype=float)
    evaluation = o3d.pipelines.registration.evaluate_registration(pcd1, pcd2, inlier_dist_threshold, trans_init)
    print(evaluation)

    draw_registration_result(pcd1, pcd2, trans_init)

    print("Initial alignment using GT")
    draw_registration_result(pcd1, pcd2, T_gt)
    evaluation = o3d.pipelines.registration.evaluate_registration(pcd1, pcd2, inlier_dist_threshold, T_gt)
    print(evaluation)

    print("Point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(pcd1, pcd2, inlier_dist_threshold, T_gt,
                                                          estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                          criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    draw_registration_result(pcd1, pcd2, reg_p2p.transformation)

    print('.')
