# Evaluate rotational invariance of a global descriptor using a joint global-local model

import argparse

import numpy as np
import tqdm
import os
import random
import pickle
from typing import List
import torch
import MinkowskiEngine as ME

from models.model_factory import model_factory
from misc.utils import ModelParams, get_datetime
from datasets.base_datasets import EvaluationTuple, EvaluationSet, get_pointcloud_loader
from datasets.quantization import Quantizer
from eval.evaluate import Evaluator
from datasets.augmentation import RandomRotation, Rotation
from models.minkgl import MinkTrunk

DEBUG = False


class MyEvaluator(Evaluator):
    # Evaluation of MinkLoc-based with MinkDD methods on Mulan or Apollo SouthBay dataset
    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str,
                 radius: List[float], k: int = 20, n_samples=None, quantizer: Quantizer = None):
        super().__init__(dataset_root, dataset_type, eval_set_pickle, device, radius, k, n_samples)
        if DEBUG:
            self.eval_set.map_set = self.eval_set.map_set[:10]
            self.eval_set.query_set = self.eval_set.query_set[:10]

        assert quantizer is not None
        self.quantizer = quantizer

    def model2eval(self, models):
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        [model.eval() for model in models]

    def evaluate(self, model, *args, **kwargs):
        map_embeddings = self.compute_embeddings(self.eval_set.map_set, model)
        global_metrics = {}
        rotations = np.arange(0., 181., 10.)

        for rotation in rotations:
            print(f'Rotation: {rotation}')
            query_embeddings = self.compute_embeddings(self.eval_set.query_set, model, rotation=rotation)

            map_positions = self.eval_set.get_map_positions()
            query_positions = self.eval_set.get_query_positions()

            if self.n_samples is None or len(query_embeddings) <= self.n_samples:
                query_indexes = list(range(len(query_embeddings)))
                self.n_samples = len(query_embeddings)
            else:
                query_indexes = random.sample(range(len(query_embeddings)), self.n_samples)

            # Dictionary to store the number of true positives (for global desc. metrics) for different radius and NN number
            global_metrics[rotation] = {'tp': {r: [0] * self.k for r in self.radius}}

            for query_ndx in tqdm.tqdm(query_indexes):
                # Check if the query element has a true match within each radius
                query_pos = query_positions[query_ndx]

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
                global_metrics[rotation]['tp'] = {r: [global_metrics[rotation]['tp'][r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in self.radius}

            # Calculate mean metrics
            global_metrics[rotation]["recall"] = {r: [global_metrics[rotation]['tp'][r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}
            print(f"Recall@1 (radius: {self.radius[0]} m.): {global_metrics[rotation]['recall'][self.radius[0]]}")

        return global_metrics

    def compute_embeddings(self, eval_subset: List[EvaluationTuple], model, *args, **kwargs):
        self.model2eval((model,))
        if 'rotation' in kwargs:
            # Rotation around z-axis
            rotation = RandomRotation(max_theta=kwargs['rotation'], max_theta2=None, axis=np.array([0, 0, 1]))
        else:
            rotation = None

        global_embeddings = None
        for ndx, e in tqdm.tqdm(enumerate(eval_subset)):
            scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
            assert os.path.exists(scan_filepath)
            pc = self.pc_loader(scan_filepath)
            if rotation is not None:
                # Rotate around z-axis
                pc = rotation(pc)

            pc = torch.tensor(pc, dtype=torch.float)
            global_embedding = self.compute_embedding(pc, model)
            if global_embeddings is None:
                global_embeddings = np.zeros((len(eval_subset), global_embedding.shape[1]), dtype=global_embedding.dtype)
            global_embeddings[ndx] = global_embedding

        return global_embeddings

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

        return global_embedding

    def print_results(self, metrics):
        # Global descriptor results are saved with the last n_k entry
        for rotation in metrics:
            print(f'Rotation: {rotation}')
            recall = metrics[rotation]['recall']
            for r in recall:
                print(f"Radius: {r} [m] : ", end='')
                for x in recall[r]:
                    print("{:0.3f}, ".format(x), end='')
                print("")
            print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate MinkLoc model')
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to the dataset root')
    parser.add_argument('--eval_set', type=str, required=True,
                        help='File name of the evaluation pickle (must be located in dataset_root')
    parser.add_argument('--radius', type=list, default=[5, 20], help='True Positive thresholds in meters')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of elements sampled from the query sequence')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the global model configuration file')
    parser.add_argument('--weights', type=str, default=None, help='Trained global model weights')

    args = parser.parse_args()
    print(f'Dataset root: {args.dataset_root}')
    print(f'Evaluation set: {args.eval_set}')
    print(f'Radius: {args.radius} [m]')
    print(f'Model config path: {args.model_config}')
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    print(f'Weights: {w}')
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

    evaluator = MyEvaluator(args.dataset_root, 'mulran', args.eval_set, device, radius=args.radius,
                            n_samples=args.n_samples, quantizer=model_params.quantizer)
    metrics = evaluator.evaluate(model)
    with open('evaluate_with_rotations_results_' + get_datetime() + '.pickle', 'wb') as f:
        pickle.dump(metrics, f)
    evaluator.print_results(metrics)
