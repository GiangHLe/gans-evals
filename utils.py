import torch
import os
import pickle
import numpy as np

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
            
def compute_feature_stats_for_dir(extractor, dataloader, only_features, cache=None, mean_cov=False, save_cache=True, name=None):
    if save_cache:
        assert name is None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor.to(device)
    extractor.eval()
    
    if cache is None:
        extent = 'feature' if only_features else 'both'
        default_path = os.path.join(os.environ['HOME'], f'/.cache/{name}_{extent}.pkl')
        if os.path.exists(default_path):
            data = load_pickle(default_path)
            return data
    else:
        data = load_pickle(cache)
        return data
                
    running_mean = None
    running_cov = None
    features = list()
    probs = list()
    
    for images in dataloader:
        assert images.shape[1] == 3
        feature, prob = extractor(images.to(device))
        if mean_cov:
            feature = feature.cpu().numpy()
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
    result = [torch.cat(features, dim=0), None, None, None]
    if not only_features:
        result[1] = torch.cat(probs, dim=0)
    if mean_cov:
        result[2] = true_mean
        result[3] = true_cov
    
    with open(default_path, 'wb') as f:
        pickle.dump(result, f)
    return result
