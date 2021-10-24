# Warsaw University of Technology

import numpy as np
from typing import List
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sklearn.neighbors import KDTree

from datasets.base_datasets import TrainingDataset, EvaluationTuple
from datasets.mulran.mulran_train import MulranTraining6DOFDataset
from datasets.augmentation import TrainTransform, TrainSetTransform
from datasets.samplers import BatchSampler
from misc.utils import TrainingParams
from datasets.base_datasets import TrainingDataset


def make_datasets(params: TrainingParams, local=False, validation=True):
    # Create training and validation datasets
    datasets = {}
    train_transform = TrainTransform(params.aug_mode)
    train_set_transform = TrainSetTransform(params.aug_mode)

    datasets['global_train'] = TrainingDataset(params.dataset_folder, params.dataset, params.train_file,
                                               transform=train_transform, set_transform=train_set_transform)
    if validation:
        datasets['global_val'] = TrainingDataset(params.dataset_folder, params.dataset, params.val_file)

    if params.secondary_dataset is not None:
        datasets['secondary_train'] = TrainingDataset(params.secondary_dataset_folder, params.secondary_dataset,
                                                      params.secondary_train_file, transform=train_transform,
                                                      set_transform=train_set_transform)

    if local:
        datasets['local_train'] = MulranTraining6DOFDataset(params.dataset_folder, params.train_file,
                                                            params.model_params.quantizer,
                                                            rot_max=params.rot_max, trans_max=params.trans_max)
        if validation:
            datasets['local_val'] = MulranTraining6DOFDataset(params.dataset_folder, params.val_file,
                                                              params.model_params.quantizer,
                                                              rot_max=params.rot_max, trans_max=params.trans_max)

    return datasets


def local_batch_to_device(batch, device):
    # Move the batch used to training the local descriptor to the proper device
    # Move everything except for len_batch
    batch['anc_batch'] = {'coords': batch['anc_batch']['coords'].to(device),
                          'features': batch['anc_batch']['features'].to(device)}
    batch['pos_batch'] = {'coords': batch['pos_batch']['coords'].to(device),
                          'features': batch['pos_batch']['features'].to(device)}
    batch['anc_pcd'] = batch['anc_pcd'].to(device)
    batch['pos_pcd'] = batch['pos_pcd'].to(device)
    batch['T_gt'] = batch['T_gt'].to(device)

    return batch


def make_collate_fn(dataset: TrainingDataset, quantizer):
    # quantizer: converts to polar (when polar coords are used) and quantizes
    def collate_fn(data_list):
        # Constructs a batch object
        clouds = [e[0] for e in data_list]
        labels = [e[1] for e in data_list]

        if dataset.set_transform is not None:
            # Apply the same transformation on all dataset elements
            lens = [len(cloud) for cloud in clouds]
            clouds = torch.cat(clouds, dim=0)
            clouds = dataset.set_transform(clouds)
            clouds = clouds.split(lens)

        # Convert to polar (when polar coords are used) and quantize
        # Use the first value returned by quantizer
        coords = [quantizer(e)[0] for e in clouds]
        coords = ME.utils.batched_coordinates(coords)

        # Assign a dummy feature equal to 1 to each point
        feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
        batch = {'coords': coords, 'features': feats}

        # Compute positives and negatives mask
        # dataset.queries[label]['positives'] is bitarray
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and negatives_mask which are
        # batch_size x batch_size boolean tensors
        #return batch, positives_mask, negatives_mask, torch.tensor(sampled_positive_ndx), torch.tensor(relative_poses)
        return batch, positives_mask, negatives_mask

    return collate_fn


def make_collate_fn_6DOF(dataset: TrainingDataset, quantizer, device):

    # ego_converted: function to convert from cartesian to polar coordinates; if None no conversion is done
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        # Constructs a batch object
        anchor_clouds = [e[0] for e in data_list]
        positive_clouds = [e[1] for e in data_list]
        rel_transforms = [e[2] for e in data_list]

        xyz_batch1, xyz_batch2 = [], []
        trans_batch, len_batch = [], []
        curr_start_inds = np.zeros((1, 2), dtype=np.int)

        for batch_id, _ in enumerate(anchor_clouds):
            N1 = anchor_clouds[batch_id].shape[0]
            N2 = positive_clouds[batch_id].shape[0]

            xyz_batch1.append(anchor_clouds[batch_id])
            xyz_batch2.append(positive_clouds[batch_id])
            trans_batch.append(rel_transforms[batch_id])

            len_batch.append([N1, N2])

            # Move the head
            curr_start_inds[0, 0] += N1
            curr_start_inds[0, 1] += N2

        anch_coords = [quantizer(e)[0] for e in anchor_clouds]
        anch_coords = ME.utils.batched_coordinates(anch_coords)
        pos_coords = [quantizer(e)[0] for e in positive_clouds]
        pos_coords = ME.utils.batched_coordinates(pos_coords)

        # Assign a dummy feature equal to 1 to each point
        anch_feats = torch.ones((anch_coords.shape[0], 1), dtype=torch.float32)
        anch_batch = {'coords': anch_coords, 'features': anch_feats}
        pos_feats = torch.ones((pos_coords.shape[0], 1), dtype=torch.float32)
        pos_batch = {'coords': pos_coords, 'features': pos_feats}

        # Concatenate point coordinates
        xyz_batch1 = torch.cat(xyz_batch1, 0).float()
        xyz_batch2 = torch.cat(xyz_batch2, 0).float()

        # Stack transforms
        trans_batch = torch.stack(trans_batch, 0).float()

        # Returns:
        # (batch_size, n_points, 3) tensor,
        # positives_mask and negatives_mask which are batch_size x batch_size boolean tensors
        # relative transformations between a pair of clouds
        return {"anc_pcd": xyz_batch1, "pos_pcd": xyz_batch2, "anc_batch": anch_batch, "pos_batch": pos_batch,
                "T_gt": trans_batch, "len_batch": len_batch}

    return collate_fn


def make_dataloaders(params: TrainingParams, debug=False, device='cpu', local=False, validation=True):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(params, local=local, validation=validation)

    dataloders = {}
    train_sampler = BatchSampler(datasets['global_train'], batch_size=params.batch_size,
                                 batch_size_limit=params.batch_size_limit,
                                 batch_expansion_rate=params.batch_expansion_rate)


    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    quantizer = params.model_params.quantizer
    train_collate_fn = make_collate_fn(datasets['global_train'],  quantizer)
    dataloders['global_train'] = DataLoader(datasets['global_train'], batch_sampler=train_sampler,
                                            collate_fn=train_collate_fn, num_workers=params.num_workers,
                                            pin_memory=True)
    if validation and 'global_val' in datasets:
        val_collate_fn = make_collate_fn(datasets['global_val'], quantizer)
        val_sampler = BatchSampler(datasets['global_val'], batch_size=params.batch_size_limit)
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        dataloders['global_val'] = DataLoader(datasets['global_val'], batch_sampler=val_sampler,
                                              collate_fn=val_collate_fn,
                                              num_workers=params.num_workers, pin_memory=True)

    if params.secondary_dataset is not None:
        secondary_train_sampler = BatchSampler(datasets['secondary_train'], batch_size=params.batch_size,
                                               batch_size_limit=params.secondary_batch_size_limit,
                                               batch_expansion_rate=params.batch_expansion_rate, max_batches=2000)

        secondary_train_collate_fn = make_collate_fn(datasets['secondary_train'],  quantizer)
        dataloders['secondary_train'] = DataLoader(datasets['secondary_train'],
                                                   batch_sampler=secondary_train_sampler,
                                                   collate_fn=secondary_train_collate_fn,
                                                   num_workers=params.num_workers,
                                                   pin_memory=True)

    if local:
        train_collate_fn_loc = make_collate_fn_6DOF(datasets['local_train'],  quantizer, device)
        val_collate_fn_loc = make_collate_fn_6DOF(datasets['local_val'], quantizer, device)
        dataloders['local_train'] = DataLoader(datasets['local_train'], shuffle=True, collate_fn=train_collate_fn_loc,
                                               batch_size=params.local_batch_size, num_workers=params.num_workers,
                                               pin_memory=True)
        if validation and 'local_val' in datasets:
            dataloders['local_val'] = DataLoader(datasets['local_val'],
                                                 collate_fn=val_collate_fn_loc, batch_size=params.local_batch_size,
                                                 num_workers=params.num_workers, pin_memory=True)

    return dataloders


def filter_query_elements(query_set: List[EvaluationTuple], map_set: List[EvaluationTuple],
                          dist_threshold: float) -> List[EvaluationTuple]:
    # Function used in evaluation dataset generation
    # Filters out query elements without a corresponding map element within dist_threshold threshold
    map_pos = np.zeros((len(map_set), 2), dtype=np.float32)
    for ndx, e in enumerate(map_set):
        map_pos[ndx] = e.position

    # Build a kdtree
    kdtree = KDTree(map_pos)

    filtered_query_set = []
    count_ignored = 0
    for ndx, e in enumerate(query_set):
        position = e.position.reshape(1, -1)
        nn = kdtree.query_radius(position, dist_threshold, count_only=True)[0]
        if nn > 0:
            filtered_query_set.append(e)
        else:
            count_ignored += 1

    print(f"{count_ignored} query elements ignored - not having corresponding map element within {dist_threshold} [m] radius")
    return filtered_query_set


def preprocess_pointcloud(pc, remove_zero_points: bool = False,
                 min_x: float = None, max_x: float = None,
                 min_y: float = None, max_y: float = None,
                 min_z: float = None, max_z: float = None):
    if remove_zero_points:
        mask = np.all(np.isclose(pc, 0.), axis=1)
        pc = pc[~mask]

    if min_x is not None:
        mask = pc[:, 0] > min_x
        pc = pc[mask]

    if max_x is not None:
        mask = pc[:, 0] <= max_x
        pc = pc[mask]

    if min_y is not None:
        mask = pc[:, 1] > min_y
        pc = pc[mask]

    if max_y is not None:
        mask = pc[:, 1] <= max_y
        pc = pc[mask]

    if min_z is not None:
        mask = pc[:, 2] > min_z
        pc = pc[mask]

    if max_z is not None:
        mask = pc[:, 2] <= max_z
        pc = pc[mask]

    return pc


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e
