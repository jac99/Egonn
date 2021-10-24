import numpy as np
import open3d as o3d
import argparse

from datasets.quantization import CartesianQuantizer
from misc.poses import apply_transform
from misc.point_clouds import draw_registration_result, draw_pc
from datasets.mulran.mulran_train import MulranTraining6DOFDataset


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
    parser = argparse.ArgumentParser(description='Evaluate MinkLoc model')
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to the dataset root')
    parser.add_argument('--train_set', type=str, default='mulran_train_tuples_Sejong01_Sejong02_10_50.pickle')
    args = parser.parse_args()

    voxel_size = 0.1
    inlier_dist_threshold = 2.      # Inlier dist threshold for ICP registration

    quantizer = CartesianQuantizer(quant_step=voxel_size)
    rot_max = np.pi
    trans_max = 5.

    dataset = MulranTraining6DOFDataset(args.dataset_root, args.train_set, quantizer,
                                        rot_max=rot_max, trans_max=trans_max)

    # ndx = 75
    ndx = 1244
    #ndx = 2410
    e = dataset[ndx]
    pc1 = e[0].numpy().astype(float)
    pc2 = e[1].numpy().astype(float)
    m = e[2].numpy().astype(float)

    pc1_transformed = apply_transform(pc1, m)
    check_alignment(pc1_transformed, pc2, inlier_dist_threshold=inlier_dist_threshold)

    dist = np.linalg.norm(m[:2, 3])
    print(f'Distance between clouds: {dist:0.2f} [m]')
    # Remove ground plane

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)

    draw_pc(pcd1)

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

    print("Initial alignment")
    draw_registration_result(pcd1, pcd2, m)
    evaluation = o3d.pipelines.registration.evaluate_registration(pcd1, pcd2, inlier_dist_threshold, m)
    print(evaluation)

    print("Point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(pcd1, pcd2, inlier_dist_threshold, m,
                                                          estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                          criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    draw_registration_result(pcd1, pcd2, reg_p2p.transformation)

    print('.')
