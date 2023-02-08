import torch
import os
import pickle
import numpy as np
from tqdm import tqdm

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
            
def compute_feature_stats_for_dir(extractor, dataloader, only_features, mean_cov=False, save_cache=True, name=None, verbose=False, to_numpy=True, float16=False, device='cpu'):
    if save_cache:
        assert name is not None

    extractor.to(device)
    extractor.eval()
    
    extent = 'feature' if only_features else 'both'
    default_cache_dir = os.path.join(os.environ['HOME'], f'.cache', 'gans_metrics')
    os.makedirs(default_cache_dir, exist_ok=True)
    default_path = os.path.join(default_cache_dir, f'{name}_{extent}_{extractor.name}.pkl')
    if os.path.exists(default_path):
        print(f'Cache exists at {default_path}, use cache')
        data = load_pickle(default_path)
        return data
                
    running_mean = None
    running_cov = None
    features = list()
    probs = list()
    loader = tqdm(dataloader) if verbose else dataloader
    
    for images in loader:
        assert images.shape[1] == 3
        prob, feature = extractor(images.to(device))
        feature = feature.cpu().numpy() if to_numpy else feature
        if float16:
            feature = feature.to(torch.float16)
        if not only_features:
            prob = prob.cpu().numpy()
        if mean_cov:
            mean = feature.sum(axis=0)
            cov = feature.T @ feature
            if running_mean is None:
                running_mean = mean
                running_cov = cov
            else:
                running_mean += mean
                running_cov += cov
        features.append(feature)
        probs.append(prob)
    if mean_cov:
        true_mean = running_mean / len(dataloader.dataset)
        true_cov = running_cov / len(dataloader.dataset)
        true_cov = true_cov - np.outer(true_mean, true_mean)
    concat = np.concatenate if to_numpy else torch.cat
    result = [concat(features, 0), None, None, None]
    if not only_features:
        result[1] = np.concatenate(probs, axis=0)
    if mean_cov:
        result[2] = true_mean
        result[3] = true_cov
    
    if save_cache:
        with open(default_path, 'wb') as f:
            pickle.dump(result, f)
    return result
