# Warsaw University of Technology

import numpy as np
import torch
from pytorch_metric_learning import losses, miners, reducers
from pytorch_metric_learning.distances import LpDistance
from misc.utils import TrainingParams
from misc.poses import apply_transform
from models.loss_utils import *

from pprint import pprint


def make_losses(params: TrainingParams):
    if params.loss == 'BatchHardTripletMarginLoss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        gl_loss_fn = BatchHardTripletLossWithMasks(params.margin)
    elif params.loss == 'BatchHardContrastiveLoss':
        gl_loss_fn = BatchHardContrastiveLossWithMasks(params.pos_margin, params.neg_margin)
    else:
        print('Unknown loss: {}'.format(params.loss))
        raise NotImplementedError

    if params.loss_gammas is not None:
       gamma_chamfer, gamma_p2p, gamma_c, beta = params.loss_gammas
    else:
        gamma_chamfer, gamma_p2p, gamma_c, beta = [1., 1., 1., 2.]
    loc_loss_fn = KeypointCorrLoss(gamma_c=gamma_c, gamma_chamfer=gamma_chamfer, gamma_p2p=gamma_p2p, beta=beta)

    return gl_loss_fn, loc_loss_fn


class KeypointCorrLoss:
    # Loss combining keypoint loss and correspondence matrix loss
    def __init__(self, gamma_c=1., gamma_k=1., gamma_chamfer=1., gamma_p2p=1., beta=1.,dist_th=0.5):
        # alpha and beta are parameters in Eq. (5) in RPM-Net paper
        self.keypoint_loss = KeypointLoss(gamma_chamfer=gamma_chamfer, gamma_p2p=gamma_p2p, prob_chamfer_loss=True,
                                          p2p_loss=True, repeatability_dist_th=dist_th)
        self.correspondence_loss = CorrespondenceLoss(beta=beta, dist_th=dist_th)

        self.gamma_k = gamma_k
        self.gamma_c = gamma_c

    def __call__(self, clouds1: torch.Tensor, keypoints1: torch.Tensor, sigma1: torch.Tensor, descriptors1: torch.Tensor,
                 clouds2: torch.Tensor, keypoints2: torch.Tensor, sigma2: torch.Tensor, descriptors2: torch.Tensor,
                 M_gt, len_batch):
        assert clouds1.dim() == 2
        assert clouds2.dim() == 2
        assert len(keypoints1) == len(sigma1)
        assert len(keypoints2) == len(sigma2)
        assert len(keypoints1) == len(descriptors1)
        assert len(keypoints2) == len(descriptors2)

        len_clouds1 = [e[0] for e in len_batch]  # Len on source point clouds
        len_clouds2 = [e[1] for e in len_batch]  # Len of target point clouds
        len_clouds1 = [0] + len_clouds1
        len_clouds2 = [0] + len_clouds2
        cum_len_clouds1 = np.cumsum(len_clouds1)    # Cummulative length
        cum_len_clouds2 = np.cumsum(len_clouds2)

        assert cum_len_clouds1[-1] == len(clouds1)
        assert cum_len_clouds2[-1] == len(clouds2)

        batch_metrics = []
        batch_loss = []

        for i, (kp1_pos, s1, desc1, kp2_pos, s2, desc2, Mi_gt) in enumerate(zip(keypoints1, sigma1, descriptors1,
                                                                                keypoints2, sigma2, descriptors2,
                                                                                M_gt)):
            pc1 = clouds1[cum_len_clouds1[i]:cum_len_clouds1[i + 1]]
            pc2 = clouds2[cum_len_clouds2[i]:cum_len_clouds2[i + 1]]

            kp1_pos_trans = apply_transform(kp1_pos, Mi_gt)
            dist_kp1_trans_kp2 = torch.cdist(kp1_pos_trans, kp2_pos)   # Euclidean distance

            metrics = {}
            metrics['kp_per_cloud'] = 0.5 * (len(kp1_pos) + len(kp2_pos))
            loss_keypoints, m = self.keypoint_loss(pc1, kp1_pos, s1, pc2, kp2_pos, s2, dist_kp1_trans_kp2)
            metrics.update(m)

            loss_correspondence, m = self.correspondence_loss(desc1, desc2, dist_kp1_trans_kp2)

            metrics.update(m)
            loss = self.gamma_k * loss_keypoints + self.gamma_c * loss_correspondence
            metrics['loss'] = loss.item()
            batch_metrics.append(metrics)
            batch_loss.append(loss)

        # Mean values per batch
        batch_loss = torch.stack(batch_loss).mean()
        batch_metrics = metrics_mean(batch_metrics)

        return batch_loss, batch_metrics


class HardTripletMinerWithMasks:
    # Hard triplet miner
    def __init__(self, distance):
        self.distance = distance
        # Stats
        self.max_pos_pair_dist = None
        self.max_neg_pair_dist = None
        self.mean_pos_pair_dist = None
        self.mean_neg_pair_dist = None
        self.min_pos_pair_dist = None
        self.min_neg_pair_dist = None

    def __call__(self, embeddings, positives_mask, negatives_mask):
        assert embeddings.dim() == 2
        d_embeddings = embeddings.detach()
        with torch.no_grad():
            hard_triplets = self.mine(d_embeddings, positives_mask, negatives_mask)
        return hard_triplets

    def mine(self, embeddings, positives_mask, negatives_mask):
        # Based on pytorch-metric-learning implementation
        dist_mat = self.distance(embeddings)
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(dist_mat, positives_mask)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(dist_mat, negatives_mask)
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(dist_mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]
        self.max_pos_pair_dist = torch.max(hardest_positive_dist).item()
        self.max_neg_pair_dist = torch.max(hardest_negative_dist).item()
        self.mean_pos_pair_dist = torch.mean(hardest_positive_dist).item()
        self.mean_neg_pair_dist = torch.mean(hardest_negative_dist).item()
        self.min_pos_pair_dist = torch.min(hardest_positive_dist).item()
        self.min_neg_pair_dist = torch.min(hardest_negative_dist).item()
        return a, p, n


def get_max_per_row(mat, mask):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat, mask):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float('inf')
    return torch.min(mat_masked, dim=1), non_inf_rows


class BatchHardTripletLossWithMasks:
    def __init__(self, margin):
        self.margin = margin
        self.distance = LpDistance(normalize_embeddings=False, collect_stats=True)
        # We use triplet loss with Euclidean distance
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        reducer_fn = reducers.MeanReducer(collect_stats=True)
        self.loss_fn = losses.TripletMarginLoss(margin=self.margin, swap=True, distance=self.distance,
                                                reducer=reducer_fn, collect_stats=True)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        pprint(dir(self.loss_fn))
        pprint(dir(self.loss_fn.reducer))

        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_non_zero_triplets': self.loss_fn.reducer.triplets_past_filter,
                 'num_triplets': len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }

        return loss, stats, hard_triplets


class BatchHardContrastiveLossWithMasks:
    def __init__(self, pos_margin, neg_margin):
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.distance = LpDistance(normalize_embeddings=False, collect_stats=True)
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        # We use contrastive loss with squared Euclidean distance
        reducer_fn = reducers.MeanReducer(collect_stats=True)
        self.loss_fn = losses.ContrastiveLoss(pos_margin=self.pos_margin, neg_margin=self.neg_margin,
                                              distance=self.distance, reducer=reducer_fn, collect_stats=True)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        stats = {'loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'pos_pairs_above_threshold': self.loss_fn.reducer.reducers['pos_loss'].pos_pairs_above_threshold,
                 'neg_pairs_above_threshold': self.loss_fn.reducer.reducers['neg_loss'].neg_pairs_above_threshold,
                 'pos_loss': self.loss_fn.reducer.reducers['pos_loss'].pos_loss.item(),
                 'neg_loss': self.loss_fn.reducer.reducers['neg_loss'].neg_loss.item(),
                 'num_pairs': 2*len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }

        return loss, stats, hard_triplets
