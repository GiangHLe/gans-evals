'''
Currently, I don't think save cache for synthesized image is reasonable, the code below is not implement that part.
'''

import os
import torch
import argparse
import json
import time
from pathlib import Path

from gans_eval.extractor import Extractor
from gans_eval.dataset import get_loader
from gans_eval.utils import compute_feature_stats_for_dir
from gans_eval.metrics import compute_is, compute_fid, compute_kid, compute_pr


MODEL_POOL = {'inception': ['inception', 299],
              'vgg': ['vgg', 224]}

def only(name, alist):
    return [name] == alist

def dump_json(name, data):
    if name is None:
        return
    with open(name, 'w') as f:
        json.dump(data, f)

def print_time(start):
    print('-----------------------------')
    print(f'Process took {round(time.time() - start, 2)}s')

def test_float16(device):
    var = torch.randn(15, 30).half().to(device)
    float16_available = True
    try:
        _ = torch.cdist(var, var)
    except:
        float16_available=False
    return float16_available
    

def main():
    opts = get_args()
    
    if opts.verbose:
        start = time.time()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    check_half = test_float16(device)    
    
    fake_dir = opts.fake_dir
    real_dir = opts.real_dir
    metrics = opts.metrics
    # assert len(os.listdir(fake_dir)) >= 50_000, "All papers recommend the number of synthesized images should be larger or equal to 50.000, \
    #     if for a specific reason, please comment this line."
    only_features = True
    json_result = {'IS': None, 'FID': None, 'KID': None, "Precision":None, 'Recall':None}
    if opts.save_json is None:
        json_path = None
        print('-----------------------------')
        print(f'Result will be printed on cmd only')
    else:
        json_path = opts.save_json    
        print('-----------------------------')
        print(f'Result will be saved at: {json_path}')
    
    mean_cov = True if 'fid' in metrics else False
    save_cache = not opts.not_save_cache
    
    if not only('pr', metrics):
        model_name = MODEL_POOL['inception'][0]
        img_size = MODEL_POOL['inception'][1]
        model = Extractor(model_name, opts.dims)
        
        # process fake data
        print('-----------------------------')
        print('Processing synthesized images...')
        fake_loader = get_loader(dataset_dir=fake_dir, 
                                 image_size=img_size, 
                                 batch_size=opts.batch_size, 
                                 num_workers=opts.num_workers,
                                 imagenet_stat=False) 
            
        if 'is' in metrics:
            only_features = False
        
        ffeatures, fprobs, fmean, fcov = compute_feature_stats_for_dir(extractor=model,
                                                                       dataloader=fake_loader,
                                                                       only_features=only_features,
                                                                       mean_cov=mean_cov,
                                                                       save_cache=False,
                                                                       name=None,
                                                                       device=device,
                                                                       verbose=opts.verbose)
        if 'is' in metrics:
            is_mean, is_std = compute_is(gen_probs = fprobs, num_splits=opts.num_splits)
            print('-----------------------------')
            print(f'IS score: {round(is_mean, 5)} +- {round(is_std, 5)}')
            json_result['IS'] = [is_mean, is_std]
            metrics.remove('is')
            if len(metrics) == 0:
                dump_json(json_path, json_result)
                if opts.verbose:
                    print_time(start)
                return

        # process real data
        print('-----------------------------')
        print('Processing real images...')
        real_loader = get_loader(dataset_dir=real_dir, 
                                image_size=img_size, 
                                batch_size=opts.batch_size, 
                                num_workers=opts.num_workers,
                                imagenet_stat=False)
        

        rfeatures, rprobs, rmean, rcov = compute_feature_stats_for_dir(extractor=model,
                                                                       dataloader=real_loader,
                                                                       only_features=only_features,
                                                                       mean_cov=mean_cov,
                                                                       save_cache=save_cache,
                                                                       name=opts.data_name,
                                                                       device=device,
                                                                       verbose=opts.verbose)
        
        if 'fid' in metrics:
            fid = compute_fid([rmean, rcov, fmean, fcov])
            print('-----------------------------')
            print(f'FID score: {round(fid, 5)}')
            json_result['FID'] = fid
            metrics.remove('fid')
            if len(metrics) == 0:
                dump_json(json_path, json_result)
                if opts.verbose:
                    print_time(start)
                return
        
        if 'kid' in metrics:
            kid = compute_kid([rfeatures, ffeatures])
            print('-----------------------------')
            print(f'KID score: {round(kid, 5)}')
            json_result['KID'] = kid
            metrics.remove('kid')
            if len(metrics) == 0:
                dump_json(json_path, json_result)
                if opts.verbose:
                    print_time(start)
                return

    model_name = MODEL_POOL['vgg'][0]
    img_size = MODEL_POOL['vgg'][1]
    model = Extractor(model_name)  
    
    print('-----------------------------')
    print('Processing synthesized images...')
    fake_loader = get_loader(dataset_dir=fake_dir, 
                            image_size=img_size, 
                            batch_size=opts.batch_size, 
                            num_workers=opts.num_workers,
                            imagenet_stat=True)
    ffeatures, _, _, _ = compute_feature_stats_for_dir(extractor=model,
                                                       dataloader=fake_loader,
                                                       only_features=True,
                                                       mean_cov=False,
                                                       save_cache=False,
                                                       name=None,
                                                       to_numpy=False,
                                                       float16=check_half,
                                                       device=device,
                                                       verbose=opts.verbose)
    
    print('-----------------------------')
    print('Processing real images...')
    real_loader = get_loader(dataset_dir=real_dir, 
                            image_size=img_size, 
                            batch_size=opts.batch_size, 
                            num_workers=opts.num_workers,
                            imagenet_stat=True)
    rfeatures, _, _, _ = compute_feature_stats_for_dir(extractor=model,
                                                       dataloader=real_loader,
                                                       only_features=True,
                                                       mean_cov=False,
                                                       save_cache=save_cache,
                                                       name=opts.data_name,
                                                       to_numpy=False,
                                                       float16=check_half,
                                                       device=device,
                                                       verbose=opts.verbose)
    precision, recall = compute_pr(rfeatures, 
                                   ffeatures, 
                                   k_nearest=opts.k_nearest, 
                                   row_batch_size=opts.row_batch_size,
                                   col_batch_size=opts.col_batch_size)
    
    json_result['Precision'] = precision
    json_result['Recall'] = recall
    dump_json(json_path, json_result)
    print('-----------------------------')
    print(f'Precision: {round(precision, 5)}, Recall: {round(recall, 5)}')  
    if opts.verbose:
        print_time(start)
    return 


def get_args():
    parser = argparse.ArgumentParser(description='GANs evaluate')
    
    parser.add_argument('--fake-dir', type=str, help='Fake image directory', required=True)
    parser.add_argument('--real-dir', default=None, type=str, help='Dataset directory, non necessary in case IS only')
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
    parser.add_argument('--dims', default = 2048, type=int, help='The position of intermediate features from Inception model, default is 2048. \n \
                                                                  Available options: \n \
                                                                    + First max pooling features: 64 \n \
                                                                    + Second max pooling features: 192 \n \
                                                                    + Pre-aux classifier features: 768 \n \
                                                                    + Final average pooling features: 2048')
    parser.add_argument('--num-workers', default = 8, type=int, help='Number of workers use for dataloader')    
    parser.add_argument('--verbose', action='store_true', help='Only apply for process bar')
    parser.add_argument('--not-save-cache', action='store_true', help='Not saving the statistic features of real dataset')

    args = parser.parse_args()
    return args

if __name__=='__main__':
    main()
    
    
    
    
    
    