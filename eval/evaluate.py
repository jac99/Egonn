# Warsaw University of Technology

import argparse

import numpy as np
import tqdm
import os
import random
from typing import List
import open3d as o3d
from time import time

import torch
import MinkowskiEngine as ME

from misc.poses import m2ypr, relative_pose
from datasets.mulran.utils import relative_pose as mulran_relative_pose
from datasets.kitti.utils import get_relative_pose as kitti_relative_pose
from datasets.quantization import Quantizer
from misc.point_clouds import icp, make_open3d_feature, make_open3d_point_cloud
from models.model_factory import model_factory
from misc.poses import apply_transform
from misc.utils import ModelParams
from datasets.dataset_utils import preprocess_pointcloud
from datasets.base_datasets import EvaluationTuple, EvaluationSet, get_pointcloud_loader


class Evaluator:
    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str,
                 radius: List[float] = (5, 20), k: int = 50, n_samples: int =None, debug: bool = False):
        # radius: list of thresholds (in meters) to consider an element from the map sequence a true positive
        # k: maximum number of nearest neighbours to consider
        # n_samples: number of samples taken from a query sequence (None=all query elements)

        assert os.path.exists(dataset_root), f"Cannot access dataset root: {dataset_root}"
        self.dataset_root = dataset_root
        self.dataset_type = dataset_type
        self.eval_set_filepath = os.path.join(dataset_root, eval_set_pickle)
        self.device = device
        self.radius = radius
        self.k = k
        self.n_samples = n_samples
        self.debug = debug

        assert os.path.exists(self.eval_set_filepath), f'Cannot access evaluation set pickle: {self.eval_set_filepath}'
        self.eval_set = EvaluationSet()
        self.eval_set.load(self.eval_set_filepath)
        if debug:
            # Make the same map set and query set in debug mdoe
            self.eval_set.map_set = self.eval_set.map_set[:4]
            self.eval_set.query_set = self.eval_set.map_set[:4]

        if n_samples is None or len(self.eval_set.query_set) <= n_samples:
            self.n_samples = len(self.eval_set.query_set)
        else:
            self.n_samples = n_samples

        self.pc_loader = get_pointcloud_loader(self.dataset_type)

    def evaluate(self, model, *args, **kwargs):
        map_embeddings = self.compute_embeddings(self.eval_set.map_set, model)
        query_embeddings = self.compute_embeddings(self.eval_set.query_set, model)

        map_positions = self.eval_set.get_map_positions()
        query_positions = self.eval_set.get_query_positions()

        # Dictionary to store the number of true positives for different radius and NN number
        tp = {r: [0] * self.k for r in self.radius}
        query_indexes = random.sample(range(len(query_embeddings)), self.n_samples)

        # Randomly sample n_samples clouds from the query sequence and NN search in the target sequence
        for query_ndx in tqdm.tqdm(query_indexes):
            # Check if the query element has a true match within each radius
            query_pos = query_positions[query_ndx]

            # Nearest neighbour search in the embedding space
            query_embedding = query_embeddings[query_ndx]
            embed_dist = np.linalg.norm(map_embeddings - query_embedding, axis=1)
            nn_ndx = np.argsort(embed_dist)[:self.k]

            # Euclidean distance between the query and nn
            delta = query_pos - map_positions[nn_ndx]  # (k, 2) array
            euclid_dist = np.linalg.norm(delta, axis=1)  # (k,) array
            # Count true positives for different radius and NN number
            tp = {r: [tp[r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in
                  self.radius}

        recall = {r: [tp[r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}
        # percentage of 'positive' queries (with at least one match in the map sequence within given radius)
        return {'recall': recall}

    def compute_embedding(self, pc, model, *args, **kwargs):
        # This method must be implemented in inheriting classes
        # Must return embedding as a numpy vector
        raise NotImplementedError('Not implemented')

    def model2eval(self, model):
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        model.eval()

    def compute_embeddings(self, eval_subset: List[EvaluationTuple], model, *args, **kwargs):
        self.model2eval(model)

        embeddings = None
        for ndx, e in tqdm.tqdm(enumerate(eval_subset)):
            scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
            assert os.path.exists(scan_filepath)
            pc = self.pc_loader(scan_filepath)
            pc = torch.tensor(pc)

            embedding = self.compute_embedding(pc, model)
            if embeddings is None:
                embeddings = np.zeros((len(eval_subset), embedding.shape[1]), dtype=embedding.dtype)
            embeddings[ndx] = embedding

        return embeddings


class MinkLocGLEvaluator(Evaluator):
    # Evaluation of EgoNN methods on Mulan or Apollo SouthBay dataset
    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str,
                 radius: List[float], k: int = 20, n_samples=None, repeat_dist_th: float = 0.5,
                 n_k: List[float] = (128, 256), quantizer: Quantizer = None,
                 icp_refine: bool = True, ignore_keypoint_saliency: bool = False, debug: bool = False):
        super().__init__(dataset_root, dataset_type, eval_set_pickle, device, radius, k, n_samples, debug=debug)
        assert quantizer is not None
        self.repeat_dist_th = repeat_dist_th
        self.quantizer = quantizer
        self.n_k = n_k
        self.icp_refine = icp_refine
        self.ignore_keypoint_saliency = ignore_keypoint_saliency

    def model2eval(self, models):
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        [model.eval() for model in models]

    def evaluate(self, model, *args, **kwargs):
        if 'only_global' in kwargs:
            self.only_global = kwargs['only_global']
        else:
            self.only_global = False

        map_embeddings, local_map_embeddings = self.compute_embeddings(self.eval_set.map_set, model)
        query_embeddings, local_query_embeddings = self.compute_embeddings(self.eval_set.query_set, model)

        map_positions = self.eval_set.get_map_positions()
        query_positions = self.eval_set.get_query_positions()

        if self.n_samples is None or len(query_embeddings) <= self.n_samples:
            query_indexes = list(range(len(query_embeddings)))
            self.n_samples = len(query_embeddings)
        else:
            query_indexes = random.sample(range(len(query_embeddings)), self.n_samples)

        if self.only_global:
            metrics = {}
        else:
            metrics = {n_kpts: {'rre': [], 'rte': [], 'repeatability': [],
                                'success': [], 'success_inliers': [], 'failure_inliers': [],
                                'rre_refined': [], 'rte_refined': [], 'success_refined': [],
                                'success_inliers_refined': [], 'repeatability_refined': [],
                                'failure_inliers_refined': [], 't_ransac': []}
                       for n_kpts in self.n_k}

        # Dictionary to store the number of true positives (for global desc. metrics) for different radius and NN number
        global_metrics = {'tp': {r: [0] * self.k for r in self.radius}}

        for query_ndx in tqdm.tqdm(query_indexes):
            # Check if the query element has a true match within each radius
            query_pos = query_positions[query_ndx]
            query_pose = self.eval_set.query_set[query_ndx].pose

            # Nearest neighbour search in the embedding space
            query_embedding = query_embeddings[query_ndx]
            embed_dist = np.linalg.norm(map_embeddings - query_embedding, axis=1)
            nn_ndx = np.argsort(embed_dist)[:self.k]

            # GLOBAL DESCRIPTOR EVALUATION
            # Euclidean distance between the query and nn
            # Here we use non-icp refined poses, but for the global descriptor it's fine
            delta = query_pos - map_positions[nn_ndx]       # (k, 2) array
            euclid_dist = np.linalg.norm(delta, axis=1)     # (k,) array
            # Count true positives for different radius and NN number
            global_metrics['tp'] = {r: [global_metrics['tp'][r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}
            if self.only_global:
                continue

            # LOCAL DESCRIPTOR EVALUATION
            # Do the evaluation only if the nn pose is within 20 meter threshold
            # Otherwise the overlap is too small to get reasonable results

            if euclid_dist[0] > 20:
                continue

            for n_kpts in self.n_k:
                # Get the first match and compute local stats only for the best match
                nn_idx = nn_ndx[0]

                # Ransac alignment
                tick = time()
                T_estimated, inliers, _ = self.ransac_fn(local_query_embeddings[query_ndx],
                                                         local_map_embeddings[nn_idx], n_kpts)

                t_ransac = time() - tick
                nn_pose = self.eval_set.map_set[nn_idx].pose
                if self.dataset_type == 'mulran':
                    T_gt = mulran_relative_pose(query_pose, nn_pose)
                elif self.dataset_type == 'southbay':
                    T_gt = relative_pose(query_pose, nn_pose)
                elif self.dataset_type == 'kitti':
                    T_gt = kitti_relative_pose(query_pose, nn_pose)
                else:
                    raise NotImplementedError('Unknown dataset type')

                # Refine the pose using ICP
                if not self.icp_refine:
                    T_refined = T_gt
                else:
                    query_filepath = os.path.join(self.dataset_root, self.eval_set.query_set[query_ndx].rel_scan_filepath)
                    query_pc = self.pc_loader(query_filepath)
                    map_filepath = os.path.join(self.dataset_root, self.eval_set.map_set[nn_idx].rel_scan_filepath)
                    map_pc = self.pc_loader(map_filepath)
                    if self.dataset_type in ['mulran', 'kitti']:
                        query_pc = preprocess_pointcloud(query_pc, remove_zero_points=True,
                                                         min_x=-80, max_x=80, min_y=-80, max_y=80, min_z=-0.9)
                        map_pc = preprocess_pointcloud(map_pc, remove_zero_points=True,
                                                         min_x=-80, max_x=80, min_y=-80, max_y=80, min_z=-0.9)
                    elif self.dataset_type == 'southbay':
                        # -1.6 removes most of the ground plane
                        query_pc = preprocess_pointcloud(query_pc, remove_zero_points=True,
                                                         min_x=-100, max_x=100, min_y=-100, max_y=100, min_z=-1.6)
                        map_pc = preprocess_pointcloud(map_pc, remove_zero_points=True,
                                                       min_x=-100, max_x=100, min_y=-100, max_y=100, min_z=-1.6)
                    else:
                        raise NotImplementedError(f"Unknown dataset type: {self.dataset_type}")
                    T_refined, _, _ = icp(query_pc, map_pc, T_gt)

                # Compute repeatability using refined pose
                kp1 = local_query_embeddings[query_ndx]['keypoints'][:n_kpts]
                kp2 = local_map_embeddings[nn_idx]['keypoints'][:n_kpts]
                metrics[n_kpts]['repeatability'].append(calculate_repeatability(kp1, kp2, T_gt, threshold=self.repeat_dist_th))
                metrics[n_kpts]['repeatability_refined'].append(calculate_repeatability(kp1, kp2, T_refined, threshold=self.repeat_dist_th))

                # calc errors
                rte = np.linalg.norm(T_estimated[:3, 3] - T_gt[:3, 3])
                cos_rre = (np.trace(T_estimated[:3, :3].transpose(1, 0) @ T_gt[:3, :3]) - 1.) / 2.
                rre = np.arccos(np.clip(cos_rre, a_min=-1., a_max=1.)) * 180. / np.pi

                metrics[n_kpts]['t_ransac'].append(t_ransac)    # RANSAC time

                # 2 meters and 5 degrees threshold for successful registration
                if rte > 2.0 or rre > 5.:
                    metrics[n_kpts]['success'].append(0.)
                    metrics[n_kpts]['failure_inliers'].append(inliers)
                else:
                    metrics[n_kpts]['success'].append(1.)
                    metrics[n_kpts]['rte'].append(rte)
                    metrics[n_kpts]['rre'].append(rre)
                    metrics[n_kpts]['success_inliers'].append(inliers)

                if self.icp_refine:
                    # calc errors using refined pose
                    rte_refined = np.linalg.norm(T_estimated[:3, 3] - T_refined[:3, 3])
                    cos_rre_refined = (np.trace(T_estimated[:3, :3].transpose(1, 0) @ T_refined[:3, :3]) - 1.) / 2.
                    rre_refined = np.arccos(np.clip(cos_rre_refined, a_min=-1., a_max=1.)) * 180. / np.pi

                    # 2 meters and 5 degrees threshold for successful registration
                    if rte_refined > 2.0 or rre_refined > 5.:
                        metrics[n_kpts]['success_refined'].append(0.)
                        metrics[n_kpts]['failure_inliers_refined'].append(inliers)
                    else:
                        metrics[n_kpts]['rte_refined'].append(rte_refined)
                        metrics[n_kpts]['rre_refined'].append(rre_refined)
                        metrics[n_kpts]['success_refined'].append(1.)
                        metrics[n_kpts]['success_inliers_refined'].append(inliers)

        # Calculate mean metrics
        global_metrics["recall"] = {r: [global_metrics['tp'][r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}

        mean_metrics = {}
        if not self.only_global:
            # Calculate mean values of local descriptor metrics
            for n_kpts in self.n_k:
                mean_metrics[n_kpts] = {}
                for metric in metrics[n_kpts]:
                    m_l = metrics[n_kpts][metric]
                    if len(m_l) == 0:
                        mean_metrics[n_kpts][metric] = 0.
                    else:
                        if metric == 't_ransac':
                            mean_metrics[n_kpts]["t_ransac_sd"] = np.std(m_l)
                        mean_metrics[n_kpts][metric] = np.mean(m_l)

        return global_metrics, mean_metrics

    def ransac_fn(self, query_keypoints, candidate_keypoints, n_k):
        """

        Returns fitness score and estimated transforms
        Estimation using Open3d 6dof ransac based on feature matching.
        """
        kp1 = query_keypoints['keypoints'][:n_k]
        kp2 = candidate_keypoints['keypoints'][:n_k]
        ransac_result = get_ransac_result(query_keypoints['features'][:n_k], candidate_keypoints['features'][:n_k],
                                          kp1, kp2)
        return ransac_result.transformation, len(ransac_result.correspondence_set), ransac_result.fitness

    def compute_embeddings(self, eval_subset: List[EvaluationTuple], model, *args, **kwargs):
        self.model2eval((model,))
        global_embeddings = None
        local_embeddings = []
        for ndx, e in tqdm.tqdm(enumerate(eval_subset)):
            scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
            assert os.path.exists(scan_filepath)
            pc = self.pc_loader(scan_filepath)
            pc = torch.tensor(pc, dtype=torch.float)

            global_embedding, keypoints, key_embeddings = self.compute_embedding(pc, model)
            if global_embeddings is None:
                global_embeddings = np.zeros((len(eval_subset), global_embedding.shape[1]), dtype=global_embedding.dtype)

            global_embeddings[ndx] = global_embedding
            local_embeddings.append({'keypoints': keypoints, 'features': key_embeddings})

        return global_embeddings, local_embeddings

    def compute_embedding(self, pc, model, *args, **kwargs):
        """
        Returns global embedding (np.array) as well as keypoints and corresponding descriptors (torch.tensors)
        """
        coords, _ = self.quantizer(pc)
        with torch.no_grad():
            bcoords = ME.utils.batched_coordinates([coords])
            feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
            batch = {'coords': bcoords.to(self.device), 'features': feats.to(self.device)}

            # Compute global descriptor
            y = model(batch)
            global_embedding = y['global'].detach().cpu().numpy()
            if 'descriptors' not in y:
                descriptors, keypoints, sigma, idxes = None, None, None, None
            else:
                descriptors, keypoints, sigma = y['descriptors'], y['keypoints'], y['sigma']
                # get sorted indexes of keypoints
                idxes = self.get_keypoints_idxes(sigma[0].squeeze(1), max(self.n_k))
                # keypoints and descriptors are in uncertainty increasing order
                keypoints = keypoints[0][idxes].detach().cpu()
                descriptors = descriptors[0][idxes].detach().cpu()

        return global_embedding, keypoints, descriptors

    def get_keypoints_idxes(self, sigmas, n_k):
        n_k = min(len(sigmas), n_k)
        if self.ignore_keypoint_saliency:
            # Get n_k random keypoints
            ndx = torch.randperm(len(sigmas))[:n_k]
        else:
            # Get n_k keypoints with the lowest uncertainty sorted in increasing order
            _, ndx = torch.topk(sigmas, dim=0, k=n_k, largest=False)

        return ndx

    def print_results(self, global_metrics, metrics):
        # Global descriptor results are saved with the last n_k entry
        recall = global_metrics['recall']
        for r in recall:
            print(f"Radius: {r} [m] : ", end='')
            for x in recall[r]:
                print("{:0.3f}, ".format(x), end='')
            print("")

        for n_kpts in self.n_k:
            if n_kpts not in metrics:
                break
            print('#keypoints: {}'.format(n_kpts))
            for s in metrics[n_kpts]:
                print(f"{s}: {metrics[n_kpts][s]:0.3f}")
            print('')


def get_ransac_result(feat1, feat2, kp1, kp2, ransac_dist_th=0.5, ransac_max_it=10000):
    feature_dim = feat1.shape[1]
    pcd_feat1 = make_open3d_feature(feat1, feature_dim, feat1.shape[0])
    pcd_feat2 = make_open3d_feature(feat2, feature_dim, feat2.shape[0])
    pcd_coord1 = make_open3d_point_cloud(kp1.numpy())
    pcd_coord2 = make_open3d_point_cloud(kp2.numpy())

    # ransac based eval
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_coord1, pcd_coord2, pcd_feat1, pcd_feat2,
        mutual_filter=True,
        max_correspondence_distance=ransac_dist_th,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_dist_th)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_max_it, 0.999))

    return ransac_result


def calculate_repeatability(kp1, kp2, T_gt, threshold: float):
    # Transform the source point cloud to the same position as the target cloud
    kp1_pos_trans = apply_transform(kp1, torch.tensor(T_gt, dtype=torch.float))
    dist = torch.cdist(kp1_pos_trans, kp2)      # (n_keypoints1, n_keypoints2) tensor

    # *** COMPUTE REPEATABILITY ***
    # Match keypoints from the first cloud with closests keypoints in the second cloud
    min_dist, _ = torch.min(dist, dim=1)
    # Repeatability with a distance threshold th
    return torch.mean((min_dist <= threshold).float()).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate MinkLoc model')
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to the dataset root')
    parser.add_argument('--dataset_type', type=str, required=True, choices=['mulran', 'southbay', 'kitti'])
    parser.add_argument('--eval_set', type=str, required=True, help='File name of the evaluation pickle (must be located in dataset_root')
    parser.add_argument('--radius', type=float, nargs='+', default=[5, 20], help='True Positive thresholds in meters')
    parser.add_argument('--n_k', type=int, nargs='+', default=[128], help='Number of keypoints to calculate repeatability')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of elements sampled from the query sequence')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the global model configuration file')
    parser.add_argument('--weights', type=str, default=None, help='Trained global model weights')
    parser.add_argument('--icp_refine', dest='icp_refine', action='store_true')
    parser.set_defaults(icp_refine=False)
    # Ignore keypoint saliency and choose keypoint randomly
    parser.add_argument('--ignore_keypoint_saliency', dest='ignore_keypoint_saliency', action='store_true')
    parser.set_defaults(ignore_keypoint_saliency=False)
    # Ignore keypoint position and assume keypoints are located at the supervoxel centres
    parser.add_argument('--ignore_keypoint_regressor', dest='ignore_keypoint_regressor', action='store_true')
    parser.set_defaults(ignore_keypoint_regressor=False)

    args = parser.parse_args()
    print(f'Dataset root: {args.dataset_root}')
    print(f'Dataset type: {args.dataset_type}')
    print(f'Evaluation set: {args.eval_set}')
    print(f'Radius: {args.radius} [m]')
    print(f'Number of keypoints for repeatability: {args.n_k}')
    print(f'Number of sampled query elements: {args.n_samples}')
    print(f'Model config path: {args.model_config}')
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print(f'Weights: {w}')
    print(f'ICP refine: {args.icp_refine}')
    print(f'Ignore keypoints saliency: {args.ignore_keypoint_saliency}')
    print(f'Ignore keypoints regressor: {args.ignore_keypoint_regressor}')
    print('')

    model_params = ModelParams(args.model_config)
    model_params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(model_params)
    if args.weights is not None:
        assert os.path.exists(args.weights), 'Cannot open network weights: {}'.format(args.weights)
        print('Loading weights: {}'.format(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)

    if args.ignore_keypoint_regressor:
        model.ignore_keypoint_regressor = True

    evaluator = MinkLocGLEvaluator(args.dataset_root, args.dataset_type, args.eval_set, device, radius=args.radius,
                                   n_samples=args.n_samples, n_k=args.n_k, quantizer=model_params.quantizer,
                                   icp_refine=args.icp_refine, ignore_keypoint_saliency=args.ignore_keypoint_saliency)
    global_metrics, metrics = evaluator.evaluate(model)
    evaluator.print_results(global_metrics, metrics)
