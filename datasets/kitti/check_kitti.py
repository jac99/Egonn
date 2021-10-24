import os
import open3d as o3d

from datasets.base_datasets import EvaluationSet, get_pointcloud_loader
from misc.point_clouds import draw_pc

if __name__ == '__main__':
    eval_set = EvaluationSet()
    dataset_root = '/media/sf_Datasets/kitti/dataset'
    eval_set_path = os.path.join(dataset_root, 'kitti_00_eval.pickle')
    eval_set.load(eval_set_path)
    pc_loader = get_pointcloud_loader('kitti')

    e = eval_set.map_set[560]
    scan_filepath = os.path.join(dataset_root, e.rel_scan_filepath)
    pc = pc_loader(scan_filepath)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc)
    pcd1 = pcd1.voxel_down_sample(voxel_size=0.3)

    draw_pc(pcd1)
