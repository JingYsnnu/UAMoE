# train_cv_uncertainty_aware_graph.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score, confusion_matrix,
                             roc_curve, precision_recall_curve)

from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
warnings.filterwarnings('ignore')


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_ece(preds, labels, n_bins=15):
    preds = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten()
    bin_edges = np.linspace(0, 1, n_bins + 1)
    n = len(preds)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (preds >= bin_edges[i]) & (preds <= bin_edges[i + 1])
        else:
            mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = preds[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def compute_aurc(preds, labels):
    order = np.argsort(-np.abs(preds - 0.5))
    correct = ((preds[order] > 0.5).astype(int) == labels[order])
    risks = 1 - np.cumsum(correct) / np.arange(1, len(correct) + 1)
    return float(risks.mean())


def compute_selective_accuracy(preds, labels, variances,
                                 coverages=(1.0, 0.8, 0.6, 0.4, 0.2)):
    order = np.argsort(variances)
    n = len(order)
    out = {}
    for cov in coverages:
        k = max(1, int(n * cov))
        idx = order[:k]
        acc = accuracy_score(labels[idx], (preds[idx] > 0.5).astype(int))
        out[f'{cov:.1f}'] = float(acc)
    return out


def compute_spearman_both(vars_, preds, labels):
    err_bin = ((preds > 0.5).astype(int) != labels.astype(int)).astype(float)
    if len(np.unique(err_bin)) > 1:
        sp_bin, p_bin = spearmanr(vars_, err_bin)
    else:
        sp_bin, p_bin = 0.0, 1.0

    err_cont = np.abs(preds - labels)
    if len(np.unique(err_cont)) > 1:
        sp_cont, p_cont = spearmanr(vars_, err_cont)
    else:
        sp_cont, p_cont = 0.0, 1.0

    return float(sp_bin), float(p_bin), float(sp_cont), float(p_cont)


class SimilarityDistribution:
    def __init__(self, matrices):
        self.num_views = len(matrices)
        self.matrices = matrices
        self.mean = np.mean(matrices, axis=0)
        self.std = np.std(matrices, axis=0)

        p95 = np.percentile(self.std, 95)
        if p95 <= 0:
            p95 = 1.0
        self.unc_norm = np.clip(self.std / (p95 + 1e-8), 0, 1)
        self.confidence = 1 - self.unc_norm

        self.multi_view_vectors = np.stack(matrices, axis=-1)


def load_data(data_path='data/'):
    CD = pd.read_csv(data_path + 'association.csv', header=None).values
    SC1 = pd.read_csv(data_path + 'gene sequence similarity.csv', header=None).values
    SC2 = pd.read_csv(data_path + 'expression_similarity_no_loop.csv', header=0,index_col=0).values
    np.fill_diagonal(SC2, 1.0)
    SD1 = pd.read_csv(data_path + 'smiles similarity.csv', header=None).values
    SD2 = pd.read_csv(data_path + 'QED-based similarity.csv', header=None).values

    for mat in [SC1, SC2, SD1, SD2]:
        mat[np.isnan(mat)] = 0
        mat[np.isinf(mat)] = 0
        np.fill_diagonal(mat, 0)

    SC1 = (SC1 + SC1.T) / 2
    SC2 = (SC2 + SC2.T) / 2
    SD1 = (SD1 + SD1.T) / 2
    SD2 = (SD2 + SD2.T) / 2

    circRNA_dist = SimilarityDistribution([SC1, SC2])
    drug_dist = SimilarityDistribution([SD1, SD2])

    return {
        'CD': CD,
        'circRNA': {
            'SC1': SC1, 'SC2': SC2,
            'distribution': circRNA_dist,
            'mean': circRNA_dist.mean,
            'std': circRNA_dist.std,
            'confidence': circRNA_dist.confidence
        },
        'drug': {
            'SD1': SD1, 'SD2': SD2,
            'distribution': drug_dist,
            'mean': drug_dist.mean,
            'std': drug_dist.std,
            'confidence': drug_dist.confidence
        }
    }


def compute_node_uncertainty_features(data):
    SC1, SC2 = data['circRNA']['SC1'], data['circRNA']['SC2']
    SD1, SD2 = data['drug']['SD1'], data['drug']['SD2']
    m, n = SC1.shape[0], SD1.shape[0]

    def compute_unc(mat1, mat2, num_nodes):
        unc = np.zeros((num_nodes, 6))
        for i in range(num_nodes):
            v1, v2 = mat1[i, :], mat2[i, :]
            unc[i, 0] = np.mean([v1.mean(), v2.mean()])
            unc[i, 1] = np.std([v1.mean(), v2.mean()])
            unc[i, 2] = np.mean(np.abs(v1 - v2))

            h1 = np.histogram(v1, bins=10)[0] + 1e-8
            h2 = np.histogram(v2, bins=10)[0] + 1e-8
            p1 = h1 / h1.sum()
            p2 = h2 / h2.sum()
            unc[i, 3] = -np.sum(p1 * np.log(p1 + 1e-8))
            unc[i, 4] = -np.sum(p2 * np.log(p2 + 1e-8))

            if np.std(v1) > 0 and np.std(v2) > 0:
                unc[i, 5] = np.corrcoef(v1, v2)[0, 1]
            else:
                unc[i, 5] = 0
        return unc

    circRNA_unc = compute_unc(SC1, SC2, m)
    drug_unc = compute_unc(SD1, SD2, n)

    scaler_c = StandardScaler()
    scaler_d = StandardScaler()
    circRNA_unc = scaler_c.fit_transform(circRNA_unc)
    drug_unc = scaler_d.fit_transform(drug_unc)

    return circRNA_unc, drug_unc


def compute_edge_stats(c, d, unc_c, unc_d, data=None, CD=None, eps=1e-8):
    c_view_mean = unc_c[c, 0]
    d_view_mean = unc_d[d, 0]
    c_view_diff = unc_c[c, 2]
    d_view_diff = unc_d[d, 2]

    unc_diff = abs(c_view_mean - d_view_mean)
    unc_mean = (c_view_mean + d_view_mean) / 2
    view_diff_sum = c_view_diff + d_view_diff

    if data is None or CD is None:
        c_corr = unc_c[c, 5]
        d_corr = unc_d[d, 5]
        return [
            unc_diff,
            unc_mean,
            view_diff_sum,
            c_view_diff * d_view_diff,
            (c_corr + d_corr) / 2,
            unc_diff / (unc_mean + eps),
        ]

    SD_mean = data['drug']['mean']
    d_neighbors_drug = np.argsort(-SD_mean[d])[:10]
    c_assoc_drug = np.where(CD[c] > 0)[0]
    overlap = len(set(d_neighbors_drug) & set(c_assoc_drug))
    overlap_norm = overlap / 10.0

    c_degree = CD[c].sum()
    d_degree = CD[:, d].sum()
    c_degree_norm = c_degree / (CD.shape[1] + eps)
    d_degree_norm = d_degree / (CD.shape[0] + eps)

    cd_assoc = float(CD[c, d])

    return [
        unc_diff,
        unc_mean,
        view_diff_sum,
        overlap_norm,
        c_degree_norm,
        d_degree_norm,
    ]


def hard_negative_mining(data, positive_pairs, num_hard=2000, random_state=42):
    np.random.seed(random_state)
    CD = data['CD']
    SC_mean = data['circRNA']['mean']
    SD_mean = data['drug']['mean']

    negative_pairs = np.argwhere(CD == 0)
    print(f"  Total negative candidates: {len(negative_pairs)}")

    c_related_drugs = {c: np.where(CD[c, :] > 0)[0] for c in range(CD.shape[0])}
    d_related_circRNAs = {d: np.where(CD[:, d] > 0)[0] for d in range(CD.shape[1])}

    difficulty = np.zeros(len(negative_pairs))
    for idx, (c, d) in enumerate(negative_pairs):
        c, d = int(c), int(d)
        score = 0.0
        c_drugs = c_related_drugs[c]
        if len(c_drugs) > 0:
            score += float(SD_mean[d, c_drugs].mean())
        d_circs = d_related_circRNAs[d]
        if len(d_circs) > 0:
            score += float(SC_mean[c, d_circs].mean())
        difficulty[idx] = score

    num_hard = min(num_hard, len(negative_pairs))
    hard_indices = np.argsort(difficulty)[-num_hard:]
    hard_negatives = negative_pairs[hard_indices]

    print(f"  Mined {len(hard_negatives)} hard negatives")
    print(f"  Difficulty range: [{difficulty.min():.4f}, {difficulty.max():.4f}]")

    return hard_negatives


def prepare_all_samples(data, random_seed=42, neg_ratio=1,
                         use_hard_negatives=True, num_hard_negatives=3000):
    set_seed(random_seed)
    CD = data['CD']
    m, n = CD.shape

    positive_pairs = np.argwhere(CD > 0)
    print(f"Positive samples: {len(positive_pairs)}")

    np.random.seed(random_seed)

    if use_hard_negatives:
        hard_negatives = hard_negative_mining(
            data, positive_pairs,
            num_hard=min(num_hard_negatives, len(positive_pairs) * 2)
        )
        all_negative_pairs = np.argwhere(CD == 0)
        hard_set = set(map(tuple, hard_negatives.tolist()))
        remaining_negatives = np.array(
            [p for p in all_negative_pairs if tuple(p.tolist()) not in hard_set]
        )

        remaining_size = max(0, len(positive_pairs) * neg_ratio - len(hard_negatives))
        if remaining_size > 0 and len(remaining_negatives) > 0:
            rand_indices = np.random.choice(
                len(remaining_negatives),
                size=min(remaining_size, len(remaining_negatives)),
                replace=False
            )
            random_negatives = remaining_negatives[rand_indices]
            negative_pairs = np.vstack([hard_negatives, random_negatives])
        else:
            negative_pairs = hard_negatives
    else:
        negative_pairs = np.argwhere(CD == 0)
        neg_sample_size = min(len(positive_pairs) * neg_ratio, len(negative_pairs))
        neg_indices = np.random.choice(len(negative_pairs), size=neg_sample_size, replace=False)
        negative_pairs = negative_pairs[neg_indices]

    print(f"Negative samples: {len(negative_pairs)}")

    all_pairs = np.vstack([positive_pairs, negative_pairs])
    all_labels = np.array([1] * len(positive_pairs) + [0] * len(negative_pairs))

    circ_conf_node = data['circRNA']['confidence'].mean(axis=1)
    drug_conf_node = data['drug']['confidence'].mean(axis=1)

    edge_confidence = np.array([
        (circ_conf_node[c] + drug_conf_node[d]) / 2
        for c, d in all_pairs
    ])

    circ_unc_c, drug_unc_d = compute_node_uncertainty_features(data)

    dist_stats = np.array([
        compute_edge_stats(c, d, circ_unc_c, drug_unc_d, data, CD)
        for c, d in all_pairs
    ])

    return {
        'all_pairs': all_pairs,
        'all_labels': all_labels,
        'edge_confidence': edge_confidence,
        'dist_stats': dist_stats,
        'unc_c': circ_unc_c,
        'unc_d': drug_unc_d,
        'num_circRNA': m,
        'num_drug': n
    }


def build_uncertainty_aware_adjacency(sim_mean, sim_std, top_k=10,
                                        sim_threshold=0.5,
                                        uncertainty_threshold=0.5,
                                        verbose=False):
    n = sim_mean.shape[0]
    adj = np.zeros_like(sim_mean)

    p95 = np.percentile(sim_std, 95)
    if p95 <= 0:
        p95 = 1.0
    sim_unc_norm = np.clip(sim_std / (p95 + 1e-8), 0, 1)

    total_sim_pass = 0
    total_after_topk = 0
    total_final = 0

    for i in range(n):
        sim_i = sim_mean[i].copy()
        unc_i = sim_unc_norm[i].copy()
        sim_i[i] = 0
        unc_i[i] = 1.0

        mask_sim = sim_i > sim_threshold
        candidates = np.where(mask_sim)[0]
        total_sim_pass += len(candidates)

        if len(candidates) == 0:
            continue

        combined = sim_i[candidates] * (1.0 - unc_i[candidates])
        if len(candidates) > top_k:
            top_k_local = np.argsort(combined)[-top_k:]
            candidates = candidates[top_k_local]
            combined = combined[top_k_local]

        total_after_topk += len(candidates)

        valid_mask = unc_i[candidates] < uncertainty_threshold
        valid_idx = candidates[valid_mask]
        valid_combined = combined[valid_mask]

        total_final += len(valid_idx)

        if len(valid_idx) == 0:
            continue

        adj[i, valid_idx] = valid_combined

    adj = (adj + adj.T) / 2

    if verbose:
        print(f"    [Adj Build] sim_th={sim_threshold}, unc_th={uncertainty_threshold}, top_k={top_k}")
        print(f"    [Adj Build] Passed sim threshold: {total_sim_pass}")
        print(f"    [Adj Build] After Top-K: {total_after_topk}")
        print(f"    [Adj Build] Final: {total_final}")
        print(f"    [Adj Build] Filter rate: {1 - total_final / max(total_after_topk, 1):.4f}")

    return adj


def build_adjacency_matrices(data, sim_threshold=0.5, uncertainty_threshold=0.5,
                               top_k=10):
    SC_mean = data['circRNA']['mean'].copy()
    SC_std = data['circRNA']['std'].copy()
    SD_mean = data['drug']['mean'].copy()
    SD_std = data['drug']['std'].copy()
    CD = data['CD'].copy()

    SC_adj = build_uncertainty_aware_adjacency(
        SC_mean, SC_std, top_k=top_k,
        sim_threshold=sim_threshold,
        uncertainty_threshold=uncertainty_threshold
    )
    SD_adj = build_uncertainty_aware_adjacency(
        SD_mean, SD_std, top_k=top_k,
        sim_threshold=sim_threshold,
        uncertainty_threshold=uncertainty_threshold
    )

    n_c = SC_mean.shape[0]
    n_d = SD_mean.shape[0]
    sc_edges = (SC_adj > 0).sum() / 2
    sd_edges = (SD_adj > 0).sum() / 2
    sc_density = sc_edges / (n_c * (n_c - 1) / 2)
    sd_density = sd_edges / (n_d * (n_d - 1) / 2)
    print(f"  [Uncertainty-aware adjacency]")
    print(f"    circRNA: {int(sc_edges)} edges, density={sc_density:.4f}")
    print(f"    drug:    {int(sd_edges)} edges, density={sd_density:.4f}")

    def normalize_adj(adj, add_self_loop=True):
        if add_self_loop:
            adj = adj + np.eye(adj.shape[0])
        degree = adj.sum(axis=1)
        degree = np.maximum(degree, 1e-8)
        degree_inv_sqrt = np.power(degree, -0.5)
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        return D_inv_sqrt @ adj @ D_inv_sqrt

    SC_norm = normalize_adj(SC_adj, add_self_loop=True)
    SD_norm = normalize_adj(SD_adj, add_self_loop=True)

    m, n = CD.shape
    CD_dense = CD.astype(float)
    c_degree = CD_dense.sum(axis=1)
    c_degree[c_degree == 0] = 1
    d_degree = CD_dense.sum(axis=0)
    d_degree[d_degree == 0] = 1
    c_inv_sqrt = np.power(c_degree, -0.5)
    d_inv_sqrt = np.power(d_degree, -0.5)
    CD_norm = np.diag(c_inv_sqrt) @ CD_dense @ np.diag(d_inv_sqrt)

    return {
        'SC_norm': torch.FloatTensor(SC_norm),
        'SD_norm': torch.FloatTensor(SD_norm),
        'CD_norm': torch.FloatTensor(CD_norm),
        'sc_density': sc_density,
        'sd_density': sd_density
    }


class GraphEncoder(nn.Module):
    def __init__(self, num_circRNA, num_drug, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.circRNA_emb = nn.Embedding(num_circRNA, hidden_dim)
        self.drug_emb = nn.Embedding(num_drug, hidden_dim)
        nn.init.xavier_uniform_(self.circRNA_emb.weight)
        nn.init.xavier_uniform_(self.drug_emb.weight)

        self.c_transforms = nn.ModuleList()
        self.d_transforms = nn.ModuleList()

        for _ in range(num_layers):
            self.c_transforms.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            self.d_transforms.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

    def forward(self, adj_SC, adj_SD, adj_CD):
        c_emb = self.circRNA_emb.weight
        d_emb = self.drug_emb.weight

        for layer in range(self.num_layers):
            c_from_c = torch.mm(adj_SC, c_emb)
            c_from_d = torch.mm(adj_CD, d_emb)
            c_agg = torch.cat([c_from_c, c_from_d], dim=-1)
            c_emb_new = self.c_transforms[layer](c_agg)

            d_from_d = torch.mm(adj_SD, d_emb)
            d_from_c = torch.mm(adj_CD.T, c_emb)
            d_agg = torch.cat([d_from_d, d_from_c], dim=-1)
            d_emb_new = self.d_transforms[layer](d_agg)

            c_emb = c_emb_new + c_emb
            d_emb = d_emb_new + d_emb

        return c_emb, d_emb


class UncertaintyAwareMoEPredictor(nn.Module):
    def __init__(self, num_circRNA, num_drug, hidden_dim=128, num_experts=4,
                 top_k=2, dropout=0.2, num_gnn_layers=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim

        self.graph_encoder = GraphEncoder(
            num_circRNA, num_drug,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout
        )

        self.unc_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dist_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        expert_input_dim = hidden_dim + hidden_dim // 2

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2)
            ) for _ in range(num_experts)
        ])

        self.gate = nn.Sequential(
            nn.Linear(expert_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )

        self.uncertainty_modulator = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, c_idx, d_idx, unc_c, unc_d, dist_stats,
                adj_SC, adj_SD, adj_CD, return_features=False):
        c_emb_all, d_emb_all = self.graph_encoder(adj_SC, adj_SD, adj_CD)
        c_graph_emb = c_emb_all[c_idx]
        d_graph_emb = d_emb_all[d_idx]
    
        unc_features = self.unc_encoder(torch.cat([unc_c, unc_d], dim=-1))
        dist_encoded = self.dist_encoder(dist_stats)
    
        combined = self.fusion(torch.cat([
            c_graph_emb, d_graph_emb, unc_features
        ], dim=-1))          
    
        expert_input = torch.cat([combined, dist_encoded], dim=-1)
    
        gate_weights = self.gate(expert_input)
        if self.top_k < self.num_experts:
            top_k_vals, top_k_idx = torch.topk(gate_weights, self.top_k, dim=-1)
            mask = torch.zeros_like(gate_weights).scatter(-1, top_k_idx, 1)
            gate_weights = gate_weights * mask
            gate_weights = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + 1e-8)
    
        expert_outputs = []
        for expert in self.experts:
            out = expert(expert_input)
            logit = out[:, 0:1]
            log_var = torch.clamp(out[:, 1:2], min=-6.0, max=0.0)
            var = torch.exp(log_var) + 1e-6
            expert_outputs.append(torch.cat([logit, var], dim=-1))
    
        expert_outputs = torch.stack(expert_outputs, dim=1)
    
        logits = expert_outputs[:, :, 0]
        vars_ = expert_outputs[:, :, 1]
    
        final_logit = (gate_weights * logits).sum(dim=-1, keepdim=True)
        final_var = (gate_weights * vars_).sum(dim=-1, keepdim=True) + \
                    0.05 * (gate_weights * (logits - final_logit).pow(2)).sum(dim=-1, keepdim=True)
    
        unc_factor = self.uncertainty_modulator(dist_stats) * 0.3
        final_var = final_var * (1 + unc_factor)
    
        final_var = 0.01 + 0.89 * torch.sigmoid(torch.log(final_var + 1e-6))
        final_var = torch.clamp(final_var, min=0.01, max=0.9)
    
        pred = torch.sigmoid(final_logit)
    
        if return_features:
            return pred, final_var, gate_weights, combined
        return pred, final_var, gate_weights


class FocalGaussianNLL(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_mu, pred_var, labels, confidences=None):
        eps = 1e-7
        pred_mu = torch.clamp(pred_mu, eps, 1 - eps)
        pred_var = torch.clamp(pred_var, min=1e-4, max=1.0)

        nll = (labels - pred_mu) ** 2 / (2 * pred_var) + 0.5 * torch.log(pred_var)

        pt = torch.where(labels == 1, pred_mu, 1 - pred_mu)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(labels == 1, self.alpha, 1 - self.alpha)

        loss = alpha_weight * focal_weight * nll

        if confidences is not None:
            conf = confidences.detach().clamp(min=0.1, max=1.0)
            loss = loss * conf

        return loss.mean()


def extract_features(model, loader, adj_SC, adj_SD, adj_CD, device='cpu'):
    model.eval()
    feats, preds, vars_, labels = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            c_idx = batch[0].to(device)
            d_idx = batch[1].to(device)
            y     = batch[2].to(device)
            dist_stats = batch[4].to(device)
            unc_c = batch[5].to(device)
            unc_d = batch[6].to(device)

            pred_mu, pred_var, _, combined = model(
                c_idx, d_idx, unc_c, unc_d, dist_stats,
                adj_SC, adj_SD, adj_CD, return_features=True)

            feats.append(combined.cpu().numpy())
            preds.append(pred_mu.squeeze().cpu().numpy())
            vars_.append(pred_var.squeeze().cpu().numpy())
            labels.append(y.squeeze().cpu().numpy())

    return (np.concatenate(feats, 0),
            np.concatenate(preds, 0),
            np.concatenate(vars_, 0),
            np.concatenate(labels, 0))


def evaluate(model, loader, adj_SC, adj_SD, adj_CD, device='cpu'):
    model.eval()
    all_preds, all_labels, all_vars = [], [], []

    with torch.no_grad():
        for batch in loader:
            c_idx = batch[0].to(device)
            d_idx = batch[1].to(device)
            labels = batch[2].to(device)
            dist_stats = batch[4].to(device)
            unc_c = batch[5].to(device)
            unc_d = batch[6].to(device)

            pred_mu, pred_var, _ = model(c_idx, d_idx, unc_c, unc_d, dist_stats,
                                          adj_SC, adj_SD, adj_CD)

            all_preds.extend(pred_mu.squeeze().cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())
            all_vars.extend(pred_var.squeeze().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_vars = np.array(all_vars)
    all_binary = (all_preds > 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    aupr = average_precision_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_binary)
    f1 = f1_score(all_labels, all_binary, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_binary, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    sp_bin, p_bin, sp_cont, p_cont = compute_spearman_both(all_vars, all_preds, all_labels)

    ece = compute_ece(all_preds, all_labels)
    aurc = compute_aurc(all_preds, all_labels)
    sel_acc = compute_selective_accuracy(all_preds, all_labels, all_vars)

    brier = float(np.mean((all_preds - all_labels) ** 2))
    nll = float(np.mean(-(all_labels * np.log(all_preds + 1e-8) +
                           (1 - all_labels) * np.log(1 - all_preds + 1e-8))))

    return {
        'auc': auc,
        'aupr': aupr,
        'acc': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'spearman': sp_bin, 'spearman_pval': p_bin,
        'spearman_cont': sp_cont, 'spearman_cont_pval': p_cont,
        'ece': ece,
        'aurc': aurc,
        'sel_acc': sel_acc,
        'brier': brier,
        'nll': nll,
        'predictions': all_preds,
        'variances': all_vars,
        'labels': all_labels,
    }


def train_epoch(model, train_loader, optimizer, criterion, adj_SC, adj_SD, adj_CD,
                device='cpu'):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(train_loader, desc='Training', leave=False):
        c_idx = batch[0].to(device)
        d_idx = batch[1].to(device)
        labels = batch[2].to(device)
        confidences = batch[3].to(device)
        dist_stats = batch[4].to(device)
        unc_c = batch[5].to(device)
        unc_d = batch[6].to(device)

        optimizer.zero_grad()

        pred_mu, pred_var, gate_weights = model(c_idx, d_idx, unc_c, unc_d,
                                                 dist_stats, adj_SC, adj_SD, adj_CD)

        if torch.isnan(pred_mu).any() or torch.isinf(pred_mu).any():
            continue

        loss = criterion(pred_mu, pred_var, labels, confidences)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def create_dataloader_from_indices(samples, indices, batch_size=256, shuffle=True):
    dataset = TensorDataset(
        torch.LongTensor(samples['all_pairs'][indices, 0]),
        torch.LongTensor(samples['all_pairs'][indices, 1]),
        torch.FloatTensor(samples['all_labels'][indices]).unsqueeze(1),
        torch.FloatTensor(samples['edge_confidence'][indices]).unsqueeze(1),
        torch.FloatTensor(samples['dist_stats'][indices]),
        torch.FloatTensor(samples['unc_c'][samples['all_pairs'][indices, 0]]),
        torch.FloatTensor(samples['unc_d'][samples['all_pairs'][indices, 1]])
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=False)


def fit_temperature(model, val_loader, adj_SC, adj_SD, adj_CD,
                     device='cpu', T_range=(0.5, 3.0), n_grid=200):
    model.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            c_idx = batch[0].to(device)
            d_idx = batch[1].to(device)
            labels = batch[2].to(device)
            dist_stats = batch[4].to(device)
            unc_c = batch[5].to(device)
            unc_d = batch[6].to(device)
            pred_mu, _, _ = model(c_idx, d_idx, unc_c, unc_d,
                                    dist_stats, adj_SC, adj_SD, adj_CD)
            pred_mu = torch.clamp(pred_mu, 1e-6, 1 - 1e-6)
            logit = torch.log(pred_mu / (1 - pred_mu))
            all_logits.append(logit.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0).flatten()
    labels = np.concatenate(all_labels, axis=0).flatten()

    T_grid = np.linspace(T_range[0], T_range[1], n_grid)
    best_T, best_score = 1.0, float('inf')

    for T in T_grid:
        scaled = 1 / (1 + np.exp(-logits / T))
        ece = compute_ece(scaled, labels)
        brier = float(np.mean((scaled - labels) ** 2))
        score = 0.5 * ece + 0.5 * brier
        if score < best_score:
            best_score = score
            best_T = T

    return float(best_T), float(best_score)


def train_single_fold(samples, train_idx, val_idx, test_idx, data, config,
                       fold_seed=42, device='cpu'):
    set_seed(fold_seed)

    adj_matrices = build_adjacency_matrices(
        data,
        sim_threshold=config.get('adj_threshold', 0.5),
        uncertainty_threshold=config.get('uncertainty_threshold', 0.5),
        top_k=config.get('top_k_neighbors', 10)
    )
    adj_SC = adj_matrices['SC_norm'].to(device)
    adj_SD = adj_matrices['SD_norm'].to(device)
    adj_CD = adj_matrices['CD_norm'].to(device)

    train_loader = create_dataloader_from_indices(
        samples, train_idx, batch_size=config['batch_size'], shuffle=True)
    val_loader = create_dataloader_from_indices(
        samples, val_idx, batch_size=config['batch_size'], shuffle=False)
    test_loader = create_dataloader_from_indices(
        samples, test_idx, batch_size=config['batch_size'], shuffle=False)

    model = UncertaintyAwareMoEPredictor(
        num_circRNA=samples['num_circRNA'],
        num_drug=samples['num_drug'],
        hidden_dim=config['hidden_dim'],
        num_experts=config['num_experts'],
        top_k=config['top_k'],
        dropout=config['dropout'],
        num_gnn_layers=config.get('num_gnn_layers', 2)
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs'],
        eta_min=config['learning_rate'] * 0.01
    )
    criterion = FocalGaussianNLL(
        alpha=config.get('focal_alpha', 0.75),
        gamma=config.get('focal_gamma', 2.0)
    )

    best_val_auc = 0
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_auc': [], 'val_aupr': [],
               'val_acc': [], 'val_spearman': [], 'val_ece': []}

    tsne_cache = {}              
    target_epochs = {1, 5, 15}   

    for epoch in range(1, config['num_epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                  adj_SC, adj_SD, adj_CD, device)

        val_metrics = evaluate(model, val_loader, adj_SC, adj_SD, adj_CD, device)

        history['train_loss'].append(train_loss)
        history['val_auc'].append(val_metrics['auc'])
        history['val_aupr'].append(val_metrics['aupr'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_spearman'].append(val_metrics['spearman'])
        history['val_ece'].append(val_metrics['ece'])

        scheduler.step()

        if epoch in target_epochs:
            feats, preds_, vars_, labels_ = extract_features(
                model, val_loader, adj_SC, adj_SD, adj_CD, device)
            tsne_cache[epoch] = (feats, preds_, vars_, labels_)

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"    Early stopping at epoch {epoch}")
                feats, preds_, vars_, labels_ = extract_features(
                    model, val_loader, adj_SC, adj_SD, adj_CD, device)
                tsne_cache[epoch] = (feats, preds_, vars_, labels_)
                break
            if epoch % 20 == 0:
                print(f"    Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                      f"Val AUC: {val_metrics['auc']:.4f} | "
                      f"Val AUPR: {val_metrics['aupr']:.4f} | "
                      f"Val Spearman: {val_metrics['spearman']:.4f} | "
                      f"Val ECE: {val_metrics['ece']:.4f}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    temperature, val_cal_score = fit_temperature(
        model, val_loader, adj_SC, adj_SD, adj_CD, device,
        T_range=(0.1, 3.0), n_grid=300)
    print(f"    [Calibration] optimal T = {temperature:.4f} "
          f"(val score = {val_cal_score:.4f})")

    test_metrics_raw = evaluate(model, test_loader,
                                 adj_SC, adj_SD, adj_CD, device)

    test_metrics = evaluate_with_temperature(
        model, test_loader, adj_SC, adj_SD, adj_CD,
        temperature=temperature, device=device)

    test_metrics['temperature'] = float(temperature)
    test_metrics['ece_raw'] = test_metrics_raw['ece']
    test_metrics['aurc_raw'] = test_metrics_raw['aurc']
    test_metrics['brier_raw'] = test_metrics_raw['brier']
    test_metrics['nll_raw'] = test_metrics_raw['nll']
    test_metrics['auc_raw'] = test_metrics_raw['auc']
    test_metrics['spearman_raw'] = test_metrics_raw['spearman']
    test_metrics['spearman_cont_raw'] = test_metrics_raw['spearman_cont']
    test_metrics['sel_acc_raw'] = test_metrics_raw['sel_acc']

    print(f"    [Debug] raw:  AUC={test_metrics_raw['auc']:.4f} | "
          f"ECE={test_metrics_raw['ece']:.4f} | "
          f"Spearman={test_metrics_raw['spearman']:.4f}")
    print(f"    [Debug] cal:  AUC={test_metrics['auc']:.4f} | "
          f"ECE={test_metrics['ece']:.4f} | "
          f"Spearman={test_metrics['spearman']:.4f} | T={temperature:.3f}")

    return model, test_metrics, history, best_val_auc, tsne_cache


def evaluate_with_temperature(model, loader, adj_SC, adj_SD, adj_CD,
                                temperature=1.0, device='cpu'):
    model.eval()
    all_preds, all_labels, all_vars = [], [], []

    with torch.no_grad():
        for batch in loader:
            c_idx = batch[0].to(device)
            d_idx = batch[1].to(device)
            labels = batch[2].to(device)
            dist_stats = batch[4].to(device)
            unc_c = batch[5].to(device)
            unc_d = batch[6].to(device)

            pred_mu, pred_var, _ = model(c_idx, d_idx, unc_c, unc_d,
                                          dist_stats, adj_SC, adj_SD, adj_CD)

            pred_mu = torch.clamp(pred_mu, 1e-6, 1 - 1e-6)
            logit = torch.log(pred_mu / (1 - pred_mu))
            calibrated = torch.sigmoid(logit / temperature)

            all_preds.extend(calibrated.squeeze().cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())
            all_vars.extend(pred_var.squeeze().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_vars = np.array(all_vars)
    all_binary = (all_preds > 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    aupr = average_precision_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_binary)
    f1 = f1_score(all_labels, all_binary, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_binary, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    sp_bin, p_bin, sp_cont, p_cont = compute_spearman_both(all_vars, all_preds, all_labels)

    ece = compute_ece(all_preds, all_labels)
    aurc = compute_aurc(all_preds, all_labels)
    sel_acc = compute_selective_accuracy(all_preds, all_labels, all_vars)
    brier = float(np.mean((all_preds - all_labels) ** 2))
    nll = float(np.mean(-(all_labels * np.log(all_preds + 1e-8) +
                           (1 - all_labels) * np.log(1 - all_preds + 1e-8))))

    return {
        'auc': auc, 'aupr': aupr, 'acc': acc, 'f1': f1,
        'precision': precision, 'recall': recall, 'specificity': specificity,
        'spearman': sp_bin, 'spearman_pval': p_bin,
        'spearman_cont': sp_cont, 'spearman_cont_pval': p_cont,
        'ece': ece, 'aurc': aurc, 'sel_acc': sel_acc,
        'brier': brier, 'nll': nll,
        'predictions': all_preds,
        'variances': all_vars,
        'labels': all_labels,
    }


def cross_validation(samples, data, config, n_folds=5, device='cpu'):
    print(f"\n{'='*60}")
    print(f"{n_folds}-Fold Cross Validation")
    print(f"{'='*60}")

    all_labels = samples['all_labels']
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    fold_models = []
    fold_histories = []
    all_test_preds = []
    all_test_labels = []
    all_test_vars = []
    all_test_circrna_idx = []
    all_test_drug_idx = []
    all_temperatures = []
    fold_tsne_caches = []

    for fold, (train_val_idx, test_idx) in enumerate(
            skf.split(np.zeros(len(all_labels)), all_labels)):
        print(f"\n{'─'*60}")
        print(f"Fold {fold+1}/{n_folds}")
        print(f"{'─'*60}")

        train_val_labels = all_labels[train_val_idx]
        train_idx, val_idx_rel = train_test_split(
            np.arange(len(train_val_idx)),
            test_size=0.15,
            random_state=42+fold,
            stratify=train_val_labels
        )
        train_idx = train_val_idx[train_idx]
        val_idx = train_val_idx[val_idx_rel]

        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        model, test_metrics, history, best_val_auc, tsne_cache = train_single_fold(
            samples, train_idx, val_idx, test_idx, data, config,
            fold_seed=42+fold, device=device
        )

        fold_results.append(test_metrics)
        fold_models.append(model)
        fold_histories.append(history)
        fold_tsne_caches.append(tsne_cache) 
        all_test_preds.extend(test_metrics['predictions'])
        all_test_labels.extend(test_metrics['labels'])
        all_test_vars.extend(test_metrics['variances'])
        all_temperatures.append(test_metrics['temperature'])
        test_pairs = samples['all_pairs'][test_idx]
        all_test_circrna_idx.extend(test_pairs[:, 0].tolist())
        all_test_drug_idx.extend(test_pairs[:, 1].tolist())

        print(f"\n  Fold {fold+1} Test Results:")
        print(f"    AUC:         {test_metrics['auc']:.4f}")
        print(f"    AUPR:        {test_metrics['aupr']:.4f}")
        print(f"    ACC:         {test_metrics['acc']:.4f}")
        print(f"    F1:          {test_metrics['f1']:.4f}")
        print(f"    Precision:   {test_metrics['precision']:.4f}")
        print(f"    Recall:      {test_metrics['recall']:.4f}")
        print(f"    Specificity: {test_metrics['specificity']:.4f}")
        print(f"    Spearman:    {test_metrics['spearman']:.4f}")
        print(f"    Spearman(c): {test_metrics['spearman_cont']:.4f}")
        print(f"    ECE:         {test_metrics['ece']:.4f}  (raw={test_metrics['ece_raw']:.4f})")
        print(f"    AURC:        {test_metrics['aurc']:.4f}  (raw={test_metrics['aurc_raw']:.4f})")
        print(f"    Brier:       {test_metrics['brier']:.4f}  (raw={test_metrics['brier_raw']:.4f})")
        print(f"    NLL:         {test_metrics['nll']:.4f}  (raw={test_metrics['nll_raw']:.4f})")
        print(f"    SelAcc@20%:  {test_metrics['sel_acc'].get('0.2', 0):.4f}")
        print(f"    T:           {test_metrics['temperature']:.4f}")

    print(f"\n{'='*60}")
    print(f"Cross Validation Summary ({n_folds} folds)")
    print(f"{'='*60}")

    metrics_to_report = [
        'auc', 'aupr', 'acc', 'f1', 'precision', 'recall', 'specificity',
        'spearman', 'spearman_cont',
        'ece', 'ece_raw',
        'aurc', 'aurc_raw',
        'brier', 'brier_raw',
        'nll', 'nll_raw',
        'temperature',
    ]

    summary = {}
    for metric in metrics_to_report:
        values = [r[metric] for r in fold_results if metric in r]
        if len(values) == 0:
            continue
        summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values,
        }
        print(f"  {metric.upper():16s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    for cov in ['1.0', '0.8', '0.6', '0.4', '0.2']:
        vals = [r['sel_acc'].get(cov, 0) for r in fold_results]
        summary[f'sel_acc_{cov}'] = {'mean': np.mean(vals), 'std': np.std(vals)}
        print(f"  SelAcc@{cov:>4s}     : {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"  {'MEAN_T':16s}: {np.mean(all_temperatures):.4f} ± "
          f"{np.std(all_temperatures):.4f}")

    all_test_preds = np.array(all_test_preds)
    all_test_labels = np.array(all_test_labels)
    all_test_vars = np.array(all_test_vars)
    all_test_binary = (all_test_preds > 0.5).astype(int)

    overall_auc = roc_auc_score(all_test_labels, all_test_preds)
    overall_aupr = average_precision_score(all_test_labels, all_test_preds)
    overall_acc = accuracy_score(all_test_labels, all_test_binary)
    overall_f1 = f1_score(all_test_labels, all_test_binary, zero_division=0)

    tn_all, fp_all, fn_all, tp_all = confusion_matrix(
        all_test_labels, all_test_binary, labels=[0, 1]).ravel()
    overall_precision = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0
    overall_recall = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0
    overall_sensitivity = overall_recall
    overall_specificity = tn_all / (tn_all + fp_all) if (tn_all + fp_all) > 0 else 0

    overall_spearman, overall_spearman_pval, \
        overall_spearman_cont, overall_spearman_cont_pval = compute_spearman_both(
        all_test_vars, all_test_preds, all_test_labels)

    overall_ece = compute_ece(all_test_preds, all_test_labels)
    overall_aurc = compute_aurc(all_test_preds, all_test_labels)
    overall_sel_acc = compute_selective_accuracy(
        all_test_preds, all_test_labels, all_test_vars,
        coverages=[1.0, 0.8, 0.6, 0.4, 0.2])
    overall_brier = float(np.mean((all_test_preds - all_test_labels) ** 2))
    overall_nll = float(np.mean(
        -(all_test_labels * np.log(all_test_preds + 1e-8) +
          (1 - all_test_labels) * np.log(1 - all_test_preds + 1e-8))))

    overall_ece_raw = np.mean([r['ece_raw'] for r in fold_results])
    overall_aurc_raw = np.mean([r['aurc_raw'] for r in fold_results])
    overall_brier_raw = np.mean([r['brier_raw'] for r in fold_results])
    overall_nll_raw = np.mean([r['nll_raw'] for r in fold_results])

    print(f"\n  Overall (all folds combined, calibrated):")
    print(f"    AUC:         {overall_auc:.4f}")
    print(f"    AUPR:        {overall_aupr:.4f}")
    print(f"    ACC:         {overall_acc:.4f}")
    print(f"    F1:          {overall_f1:.4f}")
    print(f"    Precision:   {overall_precision:.4f}")
    print(f"    Recall:      {overall_recall:.4f}")
    print(f"    Sensitivity: {overall_sensitivity:.4f}")
    print(f"    Specificity: {overall_specificity:.4f}")
    print(f"    Spearman:    {overall_spearman:.4f} (p={overall_spearman_pval:.2e})")
    print(f"    Spearman(c): {overall_spearman_cont:.4f} (p={overall_spearman_cont_pval:.2e})")
    print(f"    ECE:         {overall_ece:.4f}")
    print(f"    AURC:        {overall_aurc:.4f}")
    print(f"    Brier:       {overall_brier:.4f}")
    print(f"    NLL:         {overall_nll:.4f}")
    for cov in ['1.0', '0.8', '0.6', '0.4', '0.2']:
        print(f"    SelAcc@{cov:>4s}: {overall_sel_acc.get(cov, 0):.4f}")
    print(f"    Mean Var:    {all_test_vars.mean():.4f}")
    print(f"    Mean T:      {np.mean(all_temperatures):.4f}")

    print(f"\n  Overall (raw, uncalibrated):")
    print(f"    ECE:         {overall_ece_raw:.4f}")
    print(f"    AURC:        {overall_aurc_raw:.4f}")
    print(f"    Brier:       {overall_brier_raw:.4f}")
    print(f"    NLL:         {overall_nll_raw:.4f}")

    return {
        'fold_results': fold_results,
        'fold_models': fold_models,
        'fold_histories': fold_histories,
        'summary': summary,
        'all_test_preds': all_test_preds,
        'all_test_labels': all_test_labels,
        'all_test_vars': all_test_vars,
        'all_test_circrna_idx': all_test_circrna_idx,
        'all_test_drug_idx': all_test_drug_idx,
        'all_temperatures': all_temperatures,
        'fold_tsne_caches': fold_tsne_caches,

        'overall_auc': overall_auc,
        'overall_aupr': overall_aupr,
        'overall_acc': overall_acc,
        'overall_f1': overall_f1,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_sensitivity': overall_sensitivity,
        'overall_specificity': overall_specificity,
        'overall_spearman': overall_spearman,
        'overall_spearman_pval': overall_spearman_pval,
        'overall_spearman_cont': overall_spearman_cont,
        'overall_spearman_cont_pval': overall_spearman_cont_pval,
        'overall_ece': overall_ece,
        'overall_aurc': overall_aurc,
        'overall_sel_acc': overall_sel_acc,
        'overall_brier': overall_brier,
        'overall_nll': overall_nll,

        'overall_ece_raw': overall_ece_raw,
        'overall_aurc_raw': overall_aurc_raw,
        'overall_brier_raw': overall_brier_raw,
        'overall_nll_raw': overall_nll_raw,
    }


from sklearn.manifold import TSNE

def build_overall_tsne_cache(cv_results):
    fold_caches = cv_results.get('fold_tsne_caches', [])
    if not fold_caches:
        return None

    all_epochs = set()
    for cache in fold_caches:
        if cache:
            all_epochs.update(cache.keys())
    all_epochs = sorted(all_epochs)

    overall_cache = {}
    for ep in all_epochs:
        feats_list, preds_list, vars_list, labels_list = [], [], [], []
        for cache in fold_caches:
            if not cache or ep not in cache:
                continue
            feats, preds_, vars_, labels_ = cache[ep]
            feats_list.append(feats)
            preds_list.append(preds_)
            vars_list.append(vars_)
            labels_list.append(labels_)

        if len(feats_list) == 0:
            continue

        overall_cache[ep] = (
            np.concatenate(feats_list, axis=0),
            np.concatenate(preds_list, axis=0),
            np.concatenate(vars_list, axis=0),
            np.concatenate(labels_list, axis=0),
        )

    return overall_cache


def plot_tsne_per_fold(cv_results, save_dir):
    fold_caches = cv_results.get('fold_tsne_caches', [])
    if not fold_caches:
        print("[t-SNE] No cached features found.")
        return

    for fold_idx, cache in enumerate(fold_caches):
        if not cache:
            continue

        epochs = sorted(cache.keys())
        n_panels = len(epochs)

        if n_panels > 4:
            print(f"[t-SNE] Fold {fold_idx+1}: {n_panels} epochs > 4, "
                  f"only showing first 4.")
            epochs = epochs[:4]
            n_panels = 4

        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        axes = axes.flatten()   

        for i in range(4):
            ax = axes[i]

            if i >= n_panels:
                ax.axis('off')
                continue

            ep = epochs[i]
            feats, preds_, vars_, labels_ = cache[ep]

            n = len(feats)
            if n > 3000:
                idx = np.random.RandomState(0).choice(n, 3000, replace=False)
                feats_sub, labels_sub = feats[idx], labels_[idx]
            else:
                feats_sub, labels_sub = feats, labels_

            tsne = TSNE(n_components=2, perplexity=30,
                        learning_rate='auto', init='pca',
                        random_state=42, max_iter=1000)
            emb = tsne.fit_transform(feats_sub)

            for cls, color, name in [(0, '#d62728', 'Negative'),
                                      (1, '#1f77b4', 'Positive')]:
                m = labels_sub == cls
                ax.scatter(emb[m, 0], emb[m, 1], s=15, alpha=0.6,
                           c=color, label=name, edgecolors='none')

            ax.set_title(f'Fold {fold_idx+1} | Epoch {ep}', fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(loc='best', fontsize=12, framealpha=0.9)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'tsne_fold_{fold_idx+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[t-SNE] Saved {save_path}")
        

def plot_cv_results(cv_results, save_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    metrics = ['auc', 'aupr', 'acc', 'f1', 'spearman', 'ece']
    titles = ['AUC', 'AUPR', 'Accuracy', 'F1', 'Spearman', 'ECE']

    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        values = [r[metric] for r in cv_results['fold_results']]
        folds = [f'Fold {i+1}' for i in range(len(values))]
        bars = ax.bar(folds, values, color='steelblue', alpha=0.7)
        ax.axhline(y=np.mean(values), color='red', linestyle='--',
                   label=f'Mean = {np.mean(values):.4f}')
        ax.set_ylabel(title)
        ax.set_title(f'{title} across folds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_results.png'), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for fold, hist in enumerate(cv_results['fold_histories']):
        axes[0].plot(hist['val_auc'], label=f'Fold {fold+1}', alpha=0.7)
        axes[1].plot(hist['train_loss'], label=f'Fold {fold+1}', alpha=0.7)
        axes[2].plot(hist['val_spearman'], label=f'Fold {fold+1}', alpha=0.7)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Val AUC')
    axes[0].set_title('Validation AUC'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Train Loss')
    axes[1].set_title('Training Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Val Spearman')
    axes[2].set_title('Validation Spearman'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_training_curves.png'), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fold_aucs = [r['auc'] for r in cv_results['fold_results']]
    fold_auprs = [r['aupr'] for r in cv_results['fold_results']]
    auc_mean = np.mean(fold_aucs)
    auc_std = np.std(fold_aucs)
    aupr_mean = np.mean(fold_auprs)
    aupr_std = np.std(fold_auprs)

    for fold, result in enumerate(cv_results['fold_results']):
        fpr, tpr, _ = roc_curve(result['labels'], result['predictions'])
        axes[0].plot(fpr, tpr, alpha=0.35, linewidth=1.2, color='steelblue')

    fpr_all, tpr_all, _ = roc_curve(cv_results['all_test_labels'],
                                      cv_results['all_test_preds'])
    axes[0].plot(fpr_all, tpr_all, 'k-', linewidth=2.5,
                 label=f'AUC = {auc_mean:.4f} ± {auc_std:.4f}')

    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curves (5-Fold CV)', fontsize=13, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=11, framealpha=0.95)
    axes[0].grid(True, alpha=0.3)

    for fold, result in enumerate(cv_results['fold_results']):
        precision, recall, _ = precision_recall_curve(result['labels'],
                                                        result['predictions'])
        axes[1].plot(recall, precision, alpha=0.35, linewidth=1.2, color='steelblue')

    precision_all, recall_all, _ = precision_recall_curve(
        cv_results['all_test_labels'], cv_results['all_test_preds'])
    axes[1].plot(recall_all, precision_all, 'k-', linewidth=2.5,
                 label=f'AUPR = {aupr_mean:.4f} ± {aupr_std:.4f}')

    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('PR Curves (5-Fold CV)', fontsize=13, fontweight='bold')
    axes[1].legend(loc='lower left', fontsize=11, framealpha=0.95)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_roc_pr.png'), dpi=150)
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for fold, result in enumerate(cv_results['fold_results']):
        fpr, tpr, _ = roc_curve(result['labels'], result['predictions'])
        axes[0].plot(fpr, tpr, alpha=0.6,
                     label=f'Fold {fold+1} (AUC={result["auc"]:.4f})')
    fpr_all, tpr_all, _ = roc_curve(cv_results['all_test_labels'], cv_results['all_test_preds'])
    axes[0].plot(fpr_all, tpr_all, 'k-', linewidth=2.5,
                 label=f'Overall (AUC={cv_results["overall_auc"]:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].set_title('ROC Curves')
    axes[0].legend(loc='lower right', fontsize=9); axes[0].grid(True, alpha=0.3)

    for fold, result in enumerate(cv_results['fold_results']):
        precision, recall, _ = precision_recall_curve(result['labels'], result['predictions'])
        axes[1].plot(recall, precision, alpha=0.6,
                     label=f'Fold {fold+1} (AUPR={result["aupr"]:.4f})')
    precision_all, recall_all, _ = precision_recall_curve(cv_results['all_test_labels'],
                                                            cv_results['all_test_preds'])
    axes[1].plot(recall_all, precision_all, 'k-', linewidth=2.5,
                 label=f'Overall (AUPR={cv_results["overall_aupr"]:.4f})')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('PR Curves')
    axes[1].legend(loc='lower left', fontsize=9); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_roc_pr.png'), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(cv_results['all_test_preds'][cv_results['all_test_labels']==0],
                 bins=30, alpha=0.5, label='Negative', color='red', density=True)
    axes[0].hist(cv_results['all_test_preds'][cv_results['all_test_labels']==1],
                 bins=30, alpha=0.5, label='Positive', color='blue', density=True)
    axes[0].set_xlabel('Predicted Probability'); axes[0].set_ylabel('Density')
    axes[0].set_title('Prediction Distribution')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    scatter = axes[1].scatter(cv_results['all_test_preds'], cv_results['all_test_vars'],
                               c=cv_results['all_test_labels'], cmap='coolwarm',
                               alpha=0.5, s=10)
    axes[1].set_xlabel('Predicted Probability'); axes[1].set_ylabel('Variance')
    axes[1].set_title('Uncertainty vs Prediction')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1], label='True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_predictions.png'), dpi=150)
    plt.close()

    plot_uncertainty_binning(cv_results, save_dir)
    plot_ece_reliability(cv_results, save_dir)
    plot_risk_coverage(cv_results, save_dir)


def plot_uncertainty_binning(cv_results, save_dir):
    preds = cv_results['all_test_preds']
    vars_ = cv_results['all_test_vars']
    labels = cv_results['all_test_labels']

    n_bins = 5
    bin_edges = np.percentile(vars_, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] = bin_edges[-1] + 1e-6

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    accuracies, bin_labels, bin_sizes = [], [], []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (vars_ >= bin_edges[i]) & (vars_ <= bin_edges[i+1])
        else:
            mask = (vars_ >= bin_edges[i]) & (vars_ < bin_edges[i+1])
        if mask.sum() > 0:
            acc = accuracy_score(labels[mask], (preds[mask] > 0.5).astype(int))
            accuracies.append(acc)
            bin_sizes.append(mask.sum())
            bin_labels.append(f'Q{i+1}\n({bin_edges[i]:.3f}-\n{bin_edges[i+1]:.3f})')
        else:
            accuracies.append(0); bin_sizes.append(0)
            bin_labels.append(f'Q{i+1}\n(empty)')

    bars = axes[0].bar(range(n_bins), accuracies, color='steelblue', alpha=0.75,
                       edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_xlabel('Uncertainty Quintile (variance)', fontsize=12)
    axes[0].set_title('Accuracy vs Uncertainty Bins', fontsize=13, fontweight='bold')
    axes[0].set_ylim([0, 1.08])
    axes[0].set_xticks(range(n_bins))
    axes[0].set_xticklabels(bin_labels, fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, acc, size in zip(bars, accuracies, bin_sizes):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{acc:.4f}\n(n={size})',
                     ha='center', va='bottom', fontsize=9)

    sorted_idx = np.argsort(vars_)
    sorted_preds = preds[sorted_idx]
    sorted_labels = labels[sorted_idx]
    coverage_list = np.linspace(0.5, 1.0, 50)
    metrics_dict = {k: [] for k in ['Accuracy', 'AUC', 'F1', 'Sensitivity', 'Specificity']}

    for cov in coverage_list:
        n_use = int(len(sorted_idx) * cov)
        if n_use < 10:
            for k in metrics_dict: metrics_dict[k].append(0)
            continue
        sub_preds = sorted_preds[:n_use]
        sub_labels = sorted_labels[:n_use]
        sub_binary = (sub_preds > 0.5).astype(int)
        metrics_dict['Accuracy'].append(accuracy_score(sub_labels, sub_binary))
        if len(np.unique(sub_labels)) > 1:
            metrics_dict['AUC'].append(roc_auc_score(sub_labels, sub_preds))
        else:
            metrics_dict['AUC'].append(0)
        metrics_dict['F1'].append(f1_score(sub_labels, sub_binary, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(sub_labels, sub_binary, labels=[0, 1]).ravel()
        metrics_dict['Sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        metrics_dict['Specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    colors = {'Accuracy': 'blue', 'AUC': 'red', 'F1': 'purple',
              'Sensitivity': 'orange', 'Specificity': 'brown'}
    for name, values in metrics_dict.items():
        axes[1].plot(coverage_list, values, color=colors[name],
                     linewidth=2, label=name)
    axes[1].set_xlabel('Coverage', fontsize=12)
    axes[1].set_ylabel('Metric Value', fontsize=12)
    axes[1].set_title('Selective Prediction Performance', fontsize=13, fontweight='bold')
    axes[1].legend(loc='lower left', fontsize=9, ncol=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.5, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'uncertainty_analysis.png'), dpi=150)
    plt.close()

    print(f"\n{'='*60}")
    print("Uncertainty Binning Analysis")
    print(f"{'='*60}")
    for i, (acc, size) in enumerate(zip(accuracies, bin_sizes)):
        print(f"  Q{i+1} (var range {bin_edges[i]:.4f}-{bin_edges[i+1]:.4f}): "
              f"Acc={acc:.4f}, n={size}")


def plot_ece_reliability(cv_results, save_dir):
    preds = cv_results['all_test_preds']
    labels = cv_results['all_test_labels']
    vars_ = cv_results['all_test_vars']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    n_bins = 15
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_accs, bin_confs, bin_sizes = [], [], [], []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (preds >= bin_edges[i]) & (preds <= bin_edges[i+1])
        else:
            mask = (preds >= bin_edges[i]) & (preds < bin_edges[i+1])
        if mask.sum() == 0: continue
        bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
        bin_accs.append(labels[mask].mean())
        bin_confs.append(preds[mask].mean())
        bin_sizes.append(mask.sum())

    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect calibration')
    axes[0].bar(bin_centers, bin_accs, width=1/n_bins*0.8,
                color='steelblue', alpha=0.5, label='Empirical accuracy')
    axes[0].plot(bin_confs, bin_accs, 'r-o', markersize=5,
                 label=f'Model (ECE={cv_results["overall_ece"]:.4f})')
    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Empirical Positive Rate')
    axes[0].set_title('Reliability Diagram')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1])

    n_groups = 5
    group_edges = np.percentile(vars_, np.linspace(0, 100, n_groups + 1))
    group_edges[-1] += 1e-6
    group_ece = []
    for i in range(n_groups):
        if i == n_groups - 1:
            m = (vars_ >= group_edges[i]) & (vars_ <= group_edges[i+1])
        else:
            m = (vars_ >= group_edges[i]) & (vars_ < group_edges[i+1])
        if m.sum() < 10:
            group_ece.append(0); continue
        group_ece.append(compute_ece(preds[m], labels[m]))

    axes[1].bar(range(n_groups), group_ece, color='salmon', alpha=0.75,
                edgecolor='black')
    axes[1].set_xticks(range(n_groups))
    axes[1].set_xticklabels([f'Q{i+1}' for i in range(n_groups)])
    axes[1].set_xlabel('Uncertainty Quintile')
    axes[1].set_ylabel('ECE')
    axes[1].set_title('ECE per Uncertainty Group')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, ece in enumerate(group_ece):
        axes[1].text(i, ece + 0.005, f'{ece:.4f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ece_reliability.png'), dpi=150)
    plt.close()


def plot_risk_coverage(cv_results, save_dir):
    preds = cv_results['all_test_preds']
    labels = cv_results['all_test_labels']
    vars_ = cv_results['all_test_vars']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    coverages = np.linspace(0.05, 1.0, 50)

    for name, scores in [('Variance (ours)', -vars_),
                          ('|p-0.5| (baseline)', np.abs(preds - 0.5)),
                          ('Random', np.random.RandomState(0).randn(len(preds)))]:
        order = np.argsort(-scores)
        sorted_correct = ((preds[order] > 0.5).astype(int) == labels[order])
        risks = []
        for cov in coverages:
            n_use = max(1, int(len(order) * cov))
            risks.append(1 - sorted_correct[:n_use].mean())
        aurc = np.trapz(risks, coverages)
        axes[0].plot(coverages, risks, linewidth=2,
                     label=f'{name} (AURC={aurc:.4f})')

    axes[0].set_xlabel('Coverage')
    axes[0].set_ylabel('Risk (Error Rate)')
    axes[0].set_title('Risk-Coverage Curve')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].invert_xaxis()

    for name, scores in [('Variance (ours)', -vars_),
                          ('|p-0.5| (baseline)', np.abs(preds - 0.5)),
                          ('Random', np.random.RandomState(0).randn(len(preds)))]:
        order = np.argsort(-scores)
        accs = []
        for cov in coverages:
            n_use = max(1, int(len(order) * cov))
            idx = order[:n_use]
            accs.append(accuracy_score(labels[idx], (preds[idx] > 0.5).astype(int)))
        axes[1].plot(coverages, accs, linewidth=2, label=name)

    axes[1].set_xlabel('Coverage')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Selective Prediction Accuracy')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].invert_xaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'risk_coverage.png'), dpi=150)
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'results_uncertainty_graph_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")

    config = {
        'hidden_dim': 64,
        'num_experts': 8,
        'top_k': 4,
        'dropout': 0.3,
        'num_gnn_layers': 2,
        'learning_rate': 5e-4,
        'weight_decay': 1e-4,
        'num_epochs': 500,
        'patience': 20,
        'batch_size': 256,
        'neg_ratio': 1,
        'use_hard_negatives': True,
        'num_hard_negatives': 3000,
        'focal_alpha': 0.75,
        'focal_gamma': 2.0,
        'adj_threshold': 0, 
        'uncertainty_threshold': 0.6,
        'top_k_neighbors': 15,
        'data_path': 'data/',
        'n_folds': 5
    }

    print("\n" + "="*60)
    print("Loading Data...")
    print("="*60)
    data = load_data(config['data_path'])
    print(f"CircRNA: {data['CD'].shape[0]}, Drug: {data['CD'].shape[1]}")
    print(f"Known associations: {np.sum(data['CD'] > 0)}")

    print("\n" + "="*60)
    print("Preparing Samples (with Hard Negative Mining)...")
    print("="*60)
    samples = prepare_all_samples(
        data,
        random_seed=42,
        neg_ratio=config['neg_ratio'],
        use_hard_negatives=config['use_hard_negatives'],
        num_hard_negatives=config['num_hard_negatives']
    )
    print(f"Total samples: {len(samples['all_labels'])}")
    print(f"Positive ratio: {samples['all_labels'].mean():.4f}")

    cv_results = cross_validation(samples, data, config,
                                   n_folds=config['n_folds'], device=device)

    print(f"\n{'='*60}")
    print("Saving Results...")
    print(f"{'='*60}")

    pd.DataFrame([config]).to_csv(os.path.join(save_dir, 'config.csv'), index=False)

    summary_data = []
    for fold, result in enumerate(cv_results['fold_results']):
        row = {'fold': fold + 1}
        for k in ['auc', 'aupr', 'acc', 'f1', 'precision', 'recall',
                  'specificity', 'spearman', 'spearman_cont', 'ece', 'aurc',
                  'brier', 'nll']:
            row[k] = result[k]
        row['sel_acc_20'] = result['sel_acc'].get('0.2', 0)
        row['sel_acc_40'] = result['sel_acc'].get('0.4', 0)
        row['mean_variance'] = result['variances'].mean()
        summary_data.append(row)
    pd.DataFrame(summary_data).to_csv(os.path.join(save_dir, 'fold_results.csv'), index=False)

    summary_stats = []
    for metric in ['auc', 'aupr', 'acc', 'f1', 'spearman', 'spearman_cont',
                   'ece', 'aurc', 'brier', 'nll']:
        values = [r[metric] for r in cv_results['fold_results']]
        summary_stats.append({
            'metric': metric,
            'mean': np.mean(values), 'std': np.std(values),
            'min': np.min(values), 'max': np.max(values),
        })
    pd.DataFrame(summary_stats).to_csv(os.path.join(save_dir, 'summary.csv'), index=False)

    for fold, hist in enumerate(cv_results['fold_histories']):
        pd.DataFrame(hist).to_csv(os.path.join(save_dir, f'history_fold_{fold+1}.csv'),
                                   index=False)

    for fold, model in enumerate(cv_results['fold_models']):
        torch.save(model.state_dict(), os.path.join(save_dir, f'model_fold_{fold+1}.pt'))

    test_df = pd.DataFrame({
        'circRNA_index': cv_results['all_test_circrna_idx'],
        'drug_index':    cv_results['all_test_drug_idx'],
        'prediction':    cv_results['all_test_preds'],
        'variance':      cv_results['all_test_vars'],
        'true_label':    cv_results['all_test_labels']
    })
    test_df.to_csv(os.path.join(save_dir, 'all_test_predictions.csv'), index=False)
    print(f"Saved {len(test_df)} test predictions to all_test_predictions.csv")

    plot_cv_results(cv_results, save_dir)
    plot_tsne_per_fold(cv_results, save_dir)
    print(f"\nResults saved to: {save_dir}")

    print(f"\n{'='*60}")
    print("FINAL REPORT")
    print(f"{'='*60}")
    print(f"Model: Uncertainty-Aware MoE with Uncertainty-Aware Graph")
    print(f"Evaluation: {config['n_folds']}-Fold Cross Validation")
    print(f"\nResults (mean ± std over {config['n_folds']} folds):")
    for metric in ['auc', 'aupr', 'f1', 'spearman', 'ece', 'aurc']:
        values = [r[metric] for r in cv_results['fold_results']]
        print(f"  {metric.upper():14s}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    print(f"\nOverall (all folds combined):")
    print(f"  AUC:       {cv_results['overall_auc']:.4f}")
    print(f"  AUPR:      {cv_results['overall_aupr']:.4f}")
    print(f"  Spearman:  {cv_results['overall_spearman']:.4f}")
    print(f"  ECE:       {cv_results['overall_ece']:.4f}")
    print(f"  AURC:      {cv_results['overall_aurc']:.4f}")
    print(f"  SelAcc@20%: {cv_results['overall_sel_acc'].get('0.2', 0):.4f}")

    return cv_results


if __name__ == '__main__':
    main()