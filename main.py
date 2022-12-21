import torch
import timm
import argparse
import os
from extractor import Extractor

MODEL_POOL = {'inception': ['inception_v3', 299],
              'vgg': ['vgg16', 224]}

def main(opts):
    fake_dir = opts.fake_dir
    metrics = opts.metrics
    assert len(os.listdir(fake_dir)) >= 50_000, "All papers recommend the number of synthesized images should be larger or equal to 50.000, \
        if for a specific reason, please comment this line."
    only_features = True
    if 'fid' in metrics:
        mean_cov = True
    if 'is' in metrics:
        only_features = False
    
    
        
    
    features, probs, mean, cov = compute_feature_stats_for_dir()
    
    
        
    return None


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='GANs evaluate')
    parser.add_argument('--fake-dir', type=str, help='Fake image directory')
    parser.add_argument('--data-dir', default=None, type=str, help='Dataset directory, non necessary in case IS only')
    parser.add_argument('--metrics', default=['fid'], nargs='+', help='Options include: is, fid, kid, pr')
    parser.add_argument('--num-splits', default=10, type=int, help='The splits of Inception Score, only available if IS is selected')
    parser.add_argument('--max-real', default=1000000, type=int, help='KID parameter')
    parser.add_argument('--num-subsets', default=100, type=int, help='KID parameter')
    parser.add_argument('--max-subset-size', default=1000, type=int, help='KID parameter')
    parser.add_argument('--k-nearest', default=3, type=int, help='Precision/Recall parameter, paper recommend is 3')
    parser.add_argument('--row-batch-size', default=10000, type=int, help='Precision/Recall parameter')
    parser.add_argument('--col-batch-size', default=10000, type=int, help='Precision/Recall parameter')
    
    args = parser.parse_args()
    main(args)
    
    
    
    
    
    