# Generate training triplets

import pickle
import argparse
import numpy as np
import tqdm
import os

from datasets.southbay.southbay_raw import SouthBayDataset
from datasets.base_datasets import TrainingTuple


class Triplet:
    def __init__(self, anchor: int, positives: np.ndarray, non_negatives: np.ndarray):
        self.anchor = anchor
        self.positives = positives
        self.non_negatives = non_negatives


def generate_triplets(ds: SouthBayDataset, map_split: str, query_split: str,
                      positives_th: int = 2, negatives_th: int = 10, min_displacement: float = 0.1):
    # All elements (anchors, positives and negatives) are taken from both map_split and query_split
    assert positives_th < negatives_th

    # Create a master table with positions of all point clouds from the query split and in map_split
    pc_ids, pc_poses = ds.get_poses2([query_split, map_split])
    pc_coords = pc_poses[:, :3, 3]

    # Quantize x-y coordinates
    pos = np.floor(pc_coords / min_displacement)
    pos = pos.astype(int)
    _, unique_ndx = np.unique(pos, axis=0, return_index=True)

    # Leave only unique elements
    pc_ids = pc_ids[unique_ndx]
    pc_coords = pc_coords[unique_ndx]
    print(f'{len(pc_ids)} point clouds left from {len(pc_poses)} after filtering with min_displacement={min_displacement}')

    triplets = []
    count_zero_positives = 0
    for anchor_id in tqdm.tqdm(pc_ids):
        anchor_coords = ds.global_ndx[anchor_id].pose[:3, 3]
        dist = np.linalg.norm(pc_coords - anchor_coords, axis=1)
        positives_mask = dist <= positives_th
        non_negatives_mask = dist <= negatives_th

        positives_ndx = pc_ids[positives_mask]
        # remove anchor_id from positives
        positives_ndx = np.array([e for e in positives_ndx if e != anchor_id])
        non_negatives_ndx = pc_ids[non_negatives_mask]

        if len(positives_ndx) == 0:
            # Skip examples without positives
            count_zero_positives += 1
            continue

        t = Triplet(anchor_id, positives_ndx, non_negatives_ndx)
        triplets.append(t)

    print(f'{count_zero_positives} filtered out due to no positives')
    print(f'{len(triplets)} training tuples generated')

    # Remove ids from positives and negatives that are not anchors
    anchors_set = set([e.anchor for e in triplets])
    triplets = [Triplet(e.anchor, [p for p in e.positives if p in anchors_set],
                        [nn for nn in e.non_negatives if nn in anchors_set]) for e in triplets]

    # All used global ids
    used_ids = set()
    for triplet in triplets:
        used_ids.add(triplet.anchor)
        used_ids.update(list(triplet.positives))
        used_ids.update(list(triplet.non_negatives))

    # New ids, consecutive and starting from 0
    new_ids = {old_ndx: ndx for ndx, old_ndx in enumerate(used_ids)}

    tuples = {}
    for triplet in triplets:
        new_anchor_ndx = new_ids[triplet.anchor]
        pc = ds.global_ndx[triplet.anchor]
        positives = np.array([new_ids[e] for e in triplet.positives], dtype=np.int32)
        non_negatives = np.array([new_ids[e] for e in triplet.non_negatives], dtype=np.int32)

        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        tuple = TrainingTuple(id=new_anchor_ndx, timestamp=pc.timestamp,
                              rel_scan_filepath=pc.rel_scan_filepath,
                              positives=positives, non_negatives=non_negatives,
                              pose=pc.pose, positives_poses=None)
        tuples[new_anchor_ndx] = tuple

    return tuples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training triplets for Apollo SouthBay dataset')
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to Apollo SouthBay root folder')
    parser.add_argument('--pos_th', type=float, default=2, help='Positives threshold')
    parser.add_argument('--neg_th', type=float, default=10, help='Negatives threshold')
    parser.add_argument('--min_displacement', type=float, default=1.0)

    query_split = 'TrainData'

    args = parser.parse_args()
    print(f'Dataset root folder: {args.dataset_root}')
    print(f'Split for positives/negatives: {query_split}')
    print(f'Positives threshold: {args.pos_th}')
    print(f'Negatives threshold: {args.neg_th}')
    print(f'Minimum displacement between consecutive scans: {args.min_displacement}')

    ds = SouthBayDataset(args.dataset_root)
    ds.print_info()

    triplets = generate_triplets(ds, 'MapData', query_split, positives_th=args.pos_th, negatives_th=args.neg_th,
                                 min_displacement=args.min_displacement)
    print(f'{len(triplets)} anchors generated')

    pickle_name = f'train_southbay_{args.pos_th}_{args.neg_th}.pickle'
    pickle_filepath = os.path.join(args.dataset_root, pickle_name)
    pickle.dump(triplets, open(pickle_filepath, 'wb'))
