
import argparse
from gans_eval.metrics import Metrics
   
def main():
    opts = get_args()
    
    metrics = Metrics(**vars(opts))
    if opts.save_json is None:
        json_path = None
        print('-----------------------------')
        print(f'Result will be printed on cmd only')
    else:
        json_path = opts.save_json    
        print('-----------------------------')
        print(f'Result will be saved at: {json_path}')
    
    fake_dir = opts.fake_dir
    real_dir = opts.real_dir
    result = metrics.evaluate(fake_dir, real_dir)
    print(result)

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
    
    
    
    
    
    