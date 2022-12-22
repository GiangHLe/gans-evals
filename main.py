'''
Currently, I don't think save cache for synthesized image is reasonable, the code below is not implement that part.
'''

import torch
import timm
import argparse
import os
from extractor import Extractor
from dataset import get_loader
from utils import compute_feature_stats_for_dir
from metrics import compute_is, compute_fid, compute_kid, compute_pr
import json
from pathlib import Path

MODEL_POOL = {'inception': ['inception_v3', 299],
              'vgg': ['vgg16', 224]}

def only(name, alist):
    return [name] == alist

def dump_json(name, data):
    with open(name, 'w') as f:
        json.dump(data, f)

def main(opts):
    fake_dir = opts.fake_dir
    real_dir = opts.real_dir
    metrics = opts.metrics
    assert len(os.listdir(fake_dir)) >= 50_000, "All papers recommend the number of synthesized images should be larger or equal to 50.000, \
        if for a specific reason, please comment this line."
    only_features = True
    json_result = {'IS': None, 'FID': None, 'KID': None, "Precision":None, 'Recall':None}
    if opts.save_json is None:
        base_name = Path(opts.fake_dir).parts[-1]
        json_path = f'{base_name}.json'
    else:
        json_path = opts.save_json

    if not only('pr', metrics):
        model_name = MODEL_POOL['inception'][0]
        img_size = MODEL_POOL['inception'][1]
        model = Extractor(model_name)
        
        # process fake data
        fake_loader = get_loader(dataset_dir=fake_dir, 
                                image_size=img_size, 
                                batch_size=opts.batch_size, 
                                num_workers=opts.num_workers)
        
        if 'fid' in metrics:
            mean_cov = True
        if 'is' in metrics:
            only_features = False

        ffeatures, fprobs, fmean, fcov = compute_feature_stats_for_dir(extractor=model,
                                                                   dataloader=fake_loader,
                                                                   only_features=only_features,
                                                                   cache=None,
                                                                   mean_cov=mean_cov,
                                                                   save_cache=False,
                                                                   name=None)
        if 'is' in metrics:
            is_mean, is_std = compute_is(gen_probs = fprobs, num_splits=opts.splits)
            print(f'IS score: {round(is_mean, 3)} +- {round(is_std, 3)}')
            json_result['IS'] = [is_mean, is_std]
            metrics.remove('is')
            if len(metrics) == 0:
                dump_json(json_path, json_result)
                return

        # process real data
        real_loader = get_loader(dataset_dir=real_dir, 
                                image_size=img_size, 
                                batch_size=opts.batch_size, 
                                num_workers=opts.num_workers)
        

        rfeatures, rprobs, rmean, rcov = compute_feature_stats_for_dir(extractor=model,
                                                                       dataloader=real_loader,
                                                                       only_features=only_features,
                                                                       cache=None,
                                                                       mean_cov=mean_cov,
                                                                       save_cache=True,
                                                                       name=opts.data_name)

        if 'fid' in metrics:
            fid = compute_fid([rmean, rcov, fmean, fcov])
            json_result['FID'] = fid
            metrics.remove('fid')
            if len(metrics) == 0:
                dump_json(json_path, json_result)
                return
        
        if 'kid' in metrics:
            kid = compute_kid([rfeatures, ffeatures])
            json_result['KID'] = kid
            metrics.remove('kid')
            if len(metrics) == 0:
                dump_json(json_path, json_result)
                return

    model_name = MODEL_POOL['vgg'][0]
    img_size = MODEL_POOL['vgg'][1]
    model = Extractor(model_name)  
    fake_loader = get_loader(dataset_dir=fake_dir, 
                            image_size=img_size, 
                            batch_size=opts.batch_size, 
                            num_workers=opts.num_workers)
    ffeatures, _, _, _ = compute_feature_stats_for_dir(extractor=model,
                                                                   dataloader=fake_loader,
                                                                   only_features=only_features,
                                                                   cache=None,
                                                                   mean_cov=mean_cov,
                                                                   save_cache=False,
                                                                   name=None)

    fake_loader = get_loader(dataset_dir=real_dir, 
                            image_size=img_size, 
                            batch_size=opts.batch_size, 
                            num_workers=opts.num_workers)
    rfeatures, _, _, _ = compute_feature_stats_for_dir(extractor=model,
                                                                   dataloader=real_loader,
                                                                   only_features=True,
                                                                   cache=None,
                                                                   mean_cov=False,
                                                                   save_cache=True,
                                                                   name=opts.data_name)
    precision, recall = compute_pr(rfeatures, 
                                   ffeatures, 
                                   kth=opts.kth, 
                                   row_batch_size=opts.row_batch_size,
                                   col_batch_size=opts.col_batch_size)
    
    json_result['Precision'] = precision
    json_result['Recall'] = recall
    print(f'Precision: {round(precision, 3)}, Recall: {round(recall, 3)}')  
    return 


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='GANs evaluate')
    
    parser.add_argument('--fake-dir', type=str, help='Fake image directory')
    parser.add_argument('--data-dir', default=None, type=str, help='Dataset directory, non necessary in case IS only')
    parser.add_argument('--data-name', default='default', type=str, help='Dataset name')
    parser.add_argument('--save-json', default=None, type=str, help='where to save the result, default is working directory')
    parser.add_argument('--metrics', default=['fid'], nargs='+', help='Options include: is, fid, kid, pr')
    parser.add_argument('--num-splits', default=10, type=int, help='The splits of Inception Score, only available if IS is selected')
    parser.add_argument('--max-real', default=1000000, type=int, help='KID parameter')
    parser.add_argument('--num-subsets', default=100, type=int, help='KID parameter')
    parser.add_argument('--max-subset-size', default=1000, type=int, help='KID parameter')
    parser.add_argument('--k-nearest', default=3, type=int, help='Precision/Recall parameter, paper recommend is 3')
    parser.add_argument('--row-batch-size', default=10000, type=int, help='Precision/Recall parameter')
    parser.add_argument('--col-batch-size', default=10000, type=int, help='Precision/Recall parameter')
    parser.add_argument('--batch-size', default = 50, type=int, help='Batch size to run extractor')
    parser.add_argument('--num-workers', default = 8, type=int, help='Number of workers use for dataloader')
    

    args = parser.parse_args()
    main(args)
    
    
    
    
    
    