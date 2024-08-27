# Portions of source code adapted from the following sources:
#   https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main/metrics
#   Distributed under Apache License 2.0: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt

import os
import json
import time
import torch
import pickle
import numpy as np
import scipy.linalg


from tqdm import tqdm
from typing import List
from gans_eval.extractor import Extractor
from gans_eval.dataset import get_loader
from gans_eval.utils import load_pickle, test_float16

class Metrics():
    MODEL_POOL = {
        'inception': ['inception', 299],
        'vgg': ['vgg', 224]
    }
    
    def __init__(
        self, 
        metrics: List[str], 
        num_splits: int, 
        max_real: int = 1_000_000,
        num_subsets: int = 100,
        max_subset_size: int = 1000,
        k_nearest: int = 3, 
        row_batch_size: int = 10000, 
        col_batch_size: int = 10000, 
        mean_cov: bool = False, 
        save_cache: bool = True, 
        name: str = None, 
        verbose: bool = False, 
        batch_size: int = 50,
        num_workers: int = 9,
        dims: int = 2048,
        json_path: str = None,
        **kwargs
    ) -> None:
        self.num_splits = num_splits
        self.max_real = max_real
        self.num_subsets = num_subsets
        self.max_subset_size = max_subset_size
        self.k_nearest = k_nearest
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.mean_cov = mean_cov 
        self.save_cache = save_cache
        self.name = name
        self.verbose = verbose
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = dims
        self.save_json_path = json_path
        
        #------------------------------------------------

        self.metrics = metrics
        assert all([i for i in ('pr', 'is', 'fid', 'kid')]), "Only support metrics: pr, is ,fid, kid"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.float16 = test_float16(self.device)
    #----------------------------------------------------------------------------

    def compute_is(self, gen_probs):
        num_gen = gen_probs.shape[0]
        scores = []
        for i in range(self.num_splits):
            part = gen_probs[i * num_gen // self.num_splits : (i + 1) * num_gen // self.num_splits]
            kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
            kl = np.mean(np.sum(kl, axis=1))
            scores.append(np.exp(kl))
        return float(np.mean(scores)), float(np.std(scores))

    # -------------------------------------------------------------------------------------------------------------------------

    def compute_fid(self, stats_data):
        mu_real, sigma_real, mu_gen, sigma_gen = stats_data
        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(fid)

    # -------------------------------------------------------------------------------------------------------------------------

    def compute_kid(self, stats_data):
        real_features, gen_features = stats_data
        
        if real_features.shape[0] > self.max_real:
            only_take = np.random.choice(real_features.shape[0], self.max_real, replace=False)
            real_features = real_features[only_take]

        n = real_features.shape[1]
        m = min(min(real_features.shape[0], gen_features.shape[0]), self.max_subset_size)
        t = 0
        for _subset_idx in range(self.num_subsets):
            x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
            y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
            a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
            b = (x @ y.T / n + 1) ** 3
            t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
        kid = t / self.num_subsets / m
        return float(kid)

    # -------------------------------------------------------------------------------------------------------------------------
    
    def compute_distances(self, row_features, col_features, num_gpus, rank, col_batch_size):
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

    def compute_pr(self, real_features, gen_features):
        results = dict()
        for name, manifold, probes in [('precision', real_features, gen_features), ('recall', gen_features, real_features)]:
            kth = []
            for manifold_batch in manifold.split(self.row_batch_size):
                dist = self.compute_distances(
                    row_features=manifold_batch, 
                    col_features=manifold, 
                    col_batch_size=self.col_batch_size,
                    num_gpus=1,
                    rank=0
                )
                kth.append(dist.to(torch.float32).kthvalue(self.k_nearest + 1).values.to(torch.float16))
            kth = torch.cat(kth) 
            pred = []
            for probes_batch in probes.split(self.row_batch_size):
                dist = self.compute_distances(
                    row_features=probes_batch, 
                    col_features=manifold, 
                    col_batch_size=self.col_batch_size,
                    num_gpus=1,
                    rank=0
                )
                pred.append((dist <= kth).any(dim=1))
            results[name] = float(torch.cat(pred).to(torch.float32).mean())
        return results['precision'], results['recall']
    
    def compute_feature_stats_for_dir(self, extractor, dataloader, only_features, mean_cov, save_cache, to_numpy):
        if save_cache:
            # assert self.name is not None
            name = 'temp' if self.name is None else self.name

        extractor.to(self.device)
        extractor.eval()
        
        extent1 = 'feature' if only_features else 'both'
        extent2 = 'with_meancov' if mean_cov else 'without_meancov'
        default_cache_dir = os.path.join(os.environ['HOME'], f'.cache', 'gans_metrics')
        os.makedirs(default_cache_dir, exist_ok=True)
        default_path = os.path.join(default_cache_dir, f'{self.name}_{extent1}_{extent2}_{extractor.name}.pkl')
        if os.path.exists(default_path):
            print(f'Cache exists at {default_path}, use cache')
            data = load_pickle(default_path)
            return data
                    
        running_mean = None
        running_cov = None
        features = list()
        probs = list()
        loader = tqdm(dataloader) if self.verbose else dataloader
        
        for images in loader:
            assert images.shape[1] == 3
            prob, feature = extractor(images.to(self.device))
            feature = feature.cpu().numpy() if to_numpy else feature.cpu()
            if self.float16:
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

    def evaluate(self, fake_dir, real_dir):
        start = time.time()
        json_result = {'IS': None, 'FID': None, 'KID': None, "Precision":None, 'Recall':None}

        if 'pr' in self.metrics:
            print('-----------------------------')
            print('Computing precision and recall...')
            # load vgg
            model_name = self.MODEL_POOL['vgg'][0]
            img_size = self.MODEL_POOL['vgg'][1]
            model = Extractor(model_name)
            
            print('-----------------------------')
            print('Processing synthesized images...')
            fake_loader = get_loader(
                dataset_dir=fake_dir, 
                image_size=img_size, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers,
                imagenet_stat=True
            )

            ffeatures, _, _, _ = self.compute_feature_stats_for_dir(
                extractor=model,
                dataloader=fake_loader,
                only_features=True,
                save_cache=False,
                mean_cov=False,
                to_numpy=False
            )
            
            print('-----------------------------')
            print('Processing real images...')
            real_loader = get_loader(
                dataset_dir=real_dir, 
                image_size=img_size, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers,
                imagenet_stat=True
            )
            rfeatures, _, _, _ = self.compute_feature_stats_for_dir(
                extractor=model,
                dataloader=real_loader,
                mean_cov=False,
                only_features=True,
                save_cache=self.save_cache,
                to_numpy=False
            )
            precision, recall = self.compute_pr(
                rfeatures, 
                ffeatures, 
            )
            
            json_result['Precision'] = precision
            json_result['Recall'] = recall

            if len(self.metrics)==1:
                return json_result
            
        if len(self.metrics) > 1:
            only_features = False if 'is' in self.metrics else True
            mean_cov = True if 'fid' in self.metrics else False

            # load inception
            model_name = self.MODEL_POOL['inception'][0]
            img_size = self.MODEL_POOL['inception'][1]
            model = Extractor(model_name, self.dims)
            
            # process fake data
            print('-----------------------------')
            print('Processing synthesized images...')
            fake_loader = get_loader(
                dataset_dir=fake_dir, 
                image_size=img_size, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers,
                imagenet_stat=False
            ) 

            ffeatures, fprobs, fmean, fcov = self.compute_feature_stats_for_dir(
                extractor=model,
                dataloader=fake_loader,
                mean_cov=mean_cov,
                save_cache=False,
                only_features=only_features,
                to_numpy=True
            )

        if 'is' in self.metrics:
            is_mean, is_std = self.compute_is(gen_probs=fprobs)
            json_result['IS'] = [is_mean, is_std]

        if 'fid' in self.metrics or 'kid' in self.metrics:
            # process real data
            print('-----------------------------')
            print('Processing real images...')
            real_loader = get_loader(
                dataset_dir=real_dir, 
                image_size=img_size, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers,
                imagenet_stat=False
            )

            rfeatures, rprobs, rmean, rcov = self.compute_feature_stats_for_dir(
                extractor=model,
                dataloader=real_loader,
                only_features=only_features,
                mean_cov=mean_cov,
                save_cache=self.save_cache,
                to_numpy=True
            )

            if 'fid' in self.metrics:
                fid = self.compute_fid([rmean, rcov, fmean, fcov])
                json_result['FID'] = fid
        
            if 'kid' in self.metrics:
                kid = self.compute_kid([rfeatures, ffeatures])
                json_result['KID'] = kid
            
        if self.save_json_path:
            with open(self.save_json_path, 'w') as f:
                json.dump(json_result, f)
        
        if self.verbose:
            print('-----------------------------')
            print(f'Process took {round(time.time() - start, 2)}s')

        return json_result

            