# Portions of source code adapted from the following sources:
#   https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main/metrics
#   Distributed under Apache License 2.0: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt


import numpy as np
import scipy.linalg
import torch

#----------------------------------------------------------------------------

def compute_is(gen_probs, num_splits=10):
    num_gen = gen_probs.shape[0]
    scores = []
    for i in range(num_splits):
        part = gen_probs[i * num_gen // num_splits : (i + 1) * num_gen // num_splits]
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))

# -------------------------------------------------------------------------------------------------------------------------

def compute_fid(stats_data):
    mu_real, sigma_real, mu_gen, sigma_gen = stats_data
    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

# -------------------------------------------------------------------------------------------------------------------------

def compute_kid(stats_data,  max_real=1000000, num_subsets=100, max_subset_size=1000):
    real_features, gen_features = stats_data
    
    if real_features.shape[0] > max_real:
        only_take = np.random.choice(real_features.shape[0], max_real, replace=False)
        real_features = real_features[only_take]

    n = real_features.shape[1]
    m = min(min(real_features.shape[0], gen_features.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)

# -------------------------------------------------------------------------------------------------------------------------
 
def compute_distances(row_features, col_features, num_gpus, rank, col_batch_size):
    assert 0 <= rank < num_gpus
    num_cols = col_features.shape[0] 
    num_batches = ((num_cols - 1) // col_batch_size // num_gpus + 1) * num_gpus # 5
    col_batches = torch.nn.functional.pad(col_features, [0, 0, 0, -num_cols % num_batches]).chunk(num_batches)
    dist_batches = []
    for col_batch in col_batches[rank :: num_gpus]:
        dist_batch = torch.cdist(row_features.unsqueeze(0), col_batch.unsqueeze(0))[0]
        for src in range(num_gpus):
            dist_broadcast = dist_batch.clone()
            if num_gpus > 1:
                torch.distributed.broadcast(dist_broadcast, src=src)
            dist_batches.append(dist_broadcast.cpu() if rank == 0 else None)
    return torch.cat(dist_batches, dim=1)[:, :num_cols] if rank == 0 else None

def compute_pr(real_features, gen_features, k_nearest = 3, row_batch_size=10000, col_batch_size=10000):
    results = dict()
    for name, manifold, probes in [('precision', real_features, gen_features), ('recall', gen_features, real_features)]:
        kth = []
        for manifold_batch in manifold.split(row_batch_size):
            dist = compute_distances(row_features=manifold_batch, 
                                     col_features=manifold, 
                                     col_batch_size=col_batch_size,
                                     num_gpus=1,
                                     rank=0)
            kth.append(dist.to(torch.float32).kthvalue(k_nearest + 1).values.to(torch.float16))
        kth = torch.cat(kth) 
        pred = []
        for probes_batch in probes.split(row_batch_size):
            dist = compute_distances(row_features=probes_batch, 
                                     col_features=manifold, 
                                     col_batch_size=col_batch_size,
                                     num_gpus=1,
                                     rank=0)
            pred.append((dist <= kth).any(dim=1))
        results[name] = float(torch.cat(pred).to(torch.float32).mean())
    return results['precision'], results['recall']