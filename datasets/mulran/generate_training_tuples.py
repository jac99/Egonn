# Training tuples generation for Mulran dataset.

import numpy as np
import argparse
import tqdm
import pickle
import os

from datasets.mulran.mulran_raw import MulranSequences
from datasets.base_datasets import TrainingTuple
from datasets.mulran.utils import relative_pose
from misc.point_clouds import icp

DEBUG = False


def load_pc(file_pathname):
    # Load point cloud, clip x, y and z coords (points far away and the ground plane)
    # Returns Nx3 matrix
    pc = np.fromfile(file_pathname, dtype=np.float32)
    # PC in Mulran is of size [num_points, 4] -> x,y,z,reflectance
    pc = np.reshape(pc, (-1, 4))[:, :3]

    mask = np.all(np.isclose(pc, 0.), axis=1)
    pc = pc[~mask]
    mask = pc[:, 0] > -80
    pc = pc[mask]
    mask = pc[:, 0] <= 80

    pc = pc[mask]
    mask = pc[:, 1] > -80
    pc = pc[mask]
    mask = pc[:, 1] <= 80
    pc = pc[mask]

    mask = pc[:, 2] > -0.9
    pc = pc[mask]
    return pc


def generate_training_tuples(ds: MulranSequences, pos_threshold: float = 10, neg_threshold: float = 50):
    # displacement: displacement between consecutive anchors (if None all scans are takes as anchors).
    #               Use some small displacement to ensure there's only one scan if the vehicle does not move

    tuples = {}   # Dictionary of training tuples: tuples[ndx] = (sef ot positives, set of non negatives)
    for anchor_ndx in tqdm.tqdm(range(len(ds))):
        anchor_pos = ds.get_xy()[anchor_ndx]

        # Find timestamps of positive and negative elements
        positives = ds.find_neighbours_ndx(anchor_pos, pos_threshold)
        non_negatives = ds.find_neighbours_ndx(anchor_pos, neg_threshold)
        # Remove anchor element from positives, but leave it in non_negatives
        positives = positives[positives != anchor_ndx]

        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        # ICP pose refinement
        fitness_l = []
        inlier_rmse_l = []
        positive_poses = {}

        if DEBUG:
            # Use ground truth transform without pose refinement
            anchor_pose = ds.poses[anchor_ndx]
            for positive_ndx in positives:
                positive_pose = ds.poses[positive_ndx]
                # Compute initial relative pose
                m, fitness, inlier_rmse = relative_pose(anchor_pose, positive_pose), 1., 1.
                fitness_l.append(fitness)
                inlier_rmse_l.append(inlier_rmse)
                positive_poses[positive_ndx] = m
        else:
            anchor_pc = load_pc(os.path.join(ds.dataset_root, ds.rel_scan_filepath[anchor_ndx]))
            anchor_pose = ds.poses[anchor_ndx]
            for positive_ndx in positives:
                positive_pc = load_pc(os.path.join(ds.dataset_root, ds.rel_scan_filepath[positive_ndx]))
                positive_pose = ds.poses[positive_ndx]
                # Compute initial relative pose
                transform = relative_pose(anchor_pose, positive_pose)
                # Refine the pose using ICP
                m, fitness, inlier_rmse = icp(anchor_pc, positive_pc, transform)

                fitness_l.append(fitness)
                inlier_rmse_l.append(inlier_rmse)
                positive_poses[positive_ndx] = m

        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        tuples[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=ds.timestamps[anchor_ndx],
                                           rel_scan_filepath=ds.rel_scan_filepath[anchor_ndx],
                                           positives=positives, non_negatives=non_negatives, pose=anchor_pose,
                                           positives_poses=positive_poses)

    print(f'{len(tuples)} training tuples generated')
    print('ICP pose refimenement stats:')
    print(f'Fitness - min: {np.min(fitness_l):0.3f}   mean: {np.mean(fitness_l):0.3f}   max: {np.max(fitness_l):0.3f}')
    print(f'Inlier RMSE - min: {np.min(inlier_rmse_l):0.3f}   mean: {np.mean(inlier_rmse_l):0.3f}   max: {np.max(inlier_rmse_l):0.3f}')

    return tuples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training tuples')
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--pos_threshold', default=2)
    parser.add_argument('--neg_threshold', default=10)
    parser.add_argument('--min_displacement', type=float, default=0.2)
    args = parser.parse_args()

    sequences = ['Sejong01', 'Sejong02']
    if DEBUG:
        sequences = ['ParkingLot', 'ParkingLot']

    print(f'Dataset root: {args.dataset_root}')
    print(f'Sequences: {sequences}')
    print(f'Threshold for positive examples: {args.pos_threshold}')
    print(f'Threshold for negative examples: {args.neg_threshold}')
    print(f'Minimum displacement between consecutive anchors: {args.min_displacement}')

    ds = MulranSequences(args.dataset_root, sequences, split='train', min_displacement=args.min_displacement)
    train_tuples = generate_training_tuples(ds, args.pos_threshold, args.neg_threshold)
    pickle_name = f'train_{sequences[0]}_{sequences[1]}_{args.pos_threshold}_{args.neg_threshold}.pickle'
    train_tuples_filepath = os.path.join(args.dataset_root, pickle_name)
    pickle.dump(train_tuples, open(train_tuples_filepath, 'wb'))
    train_tuples = None

    ds = MulranSequences(args.dataset_root, sequences, split='test', min_displacement=args.min_displacement)
    test_tuples = generate_training_tuples(ds, args.pos_threshold, args.neg_threshold)
    pickle_name = f'val_{sequences[0]}_{sequences[1]}_{args.pos_threshold}_{args.neg_threshold}.pickle'
    test_tuples_filepath = os.path.join(args.dataset_root, pickle_name)
    pickle.dump(test_tuples, open(test_tuples_filepath, 'wb'))
