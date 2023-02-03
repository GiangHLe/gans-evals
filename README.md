# GANs evaluation metrics Pytorch version  

This is the implementation of all current state-of-the-art GANs evaluation metrics which include: Inception Score (IS), Fréchet Inception Distance (FID), Kernel Inception Distance (KID), Precision and Recall (PR).

Although there are quite a lot of implementations of those metrics in other repo, their usage and requirements are not the same which may cause inconvenience during the process. This package's target is to wrap all metrics into one setting and simplify the usage.

[Inception Score (IS)](https://arxiv.org/abs/1606.03498) is a metric to judge the reality and the variety of synthesis images based on the [Kullback-Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between the probability outputs of [inception model](https://arxiv.org/abs/1512.00567) of each fake image and the marginal distribution (the distribution cummulation of all samples). The original implementation by Tensorflow could be found [here](https://github.com/openai/improved-gan).

[Fréchet Inception Distance (FID)](https://arxiv.org/abs/1706.08500) compares the [Fréchet distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance) between two datasets by the mean and covariance of each dataset features. Features are extracted from the last average pooling layer of the [inception model](https://arxiv.org/abs/1512.00567). The original implementation by Tensorflow could be found [here](https://github.com/bioinf-jku/TTUR).

[Kernel Inception Distance (KID)](https://arxiv.org/pdf/1801.01401) compares the Maximum Mean Discrepancy (MMD) between two datasets by dataset features. Features are extracted using the same method as FID. The original implementation by Tensorflow could be found [here](https://github.com/mbinkowski/MMD-GAN).

[Precision and Recall (PR)](https://arxiv.org/abs/1904.06991) of GANs could be understood as the quality and diversity of synthesis images, respectively. The method uses [VGG16](https://arxiv.org/abs/1409.1556) as feature extractor, the real and fake domain then decided by k-nearest neighor in eucidean space, precision and recall value is computed by the intersection between two domains. The original implementation by Tensorflow could be found [here](https://github.com/kynkaat/improved-precision-and-recall-metric).

The Inception model and FID implementation is modified from [the unofficial Pytorch FID implementation](https://github.com/mseitzer/pytorch-fid). The VGG model uses [timm package](https://pypi.org/project/timm/) and the implementation of IS, FID and PR is modified from [Pytorch StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch).

# Installation

Install from pip:

>pip install gans-eval


Requirements:

+ numpy
+ opencv-python
+ pillow
+ scipy
+ tqdm
+ timm
+ torch>=1.7.0
+ torchvision>=0.8.0

# Usage

Except for IS, other metrics need two individual datasets to process, the simple usage:

```
python main.py --fake-dir path_to_syn_dataset --real-dir path_to_real_dataset --metrics is fid kid pr
```

All available arguments:

+ `--fake-dir` Synthesis dataset directory.

+ `--real-dir` Real dataset directory.

+ `--data-name` the statistic features of dataset will be save as pickle file in `['HOME']/.cache/gans_metrics/dataset_name.pkl`. Default: `default`.

+ `--save-json` path to save result as json file. Default: `None`.

+ `--metrics` evaluation method. Available options `is, fid, kid, pr`. Default: `fid`.

+ `--num-splits` parameter to calculate IS, number of divisions to calculate statistics. This parameter together with the number of samples can cause the IS to change. Default: `10`.

+ `--max-real` parameter to calculate KID, the maximum number of real samples. Default: `1000000`.

+ `--num-subsets` parameter to calculate KID, the number of subsets while compute KID. Default: `100`.

+ `--max-subset-size` parameter to calculate KID, the maximum number of sample in one subset. Default: `1000`.

+ `--k-nearest` parameter to calculate PR, the kth nearest neighbor. Default: `3`.

+ `--row-batch-size` parameter to calculate PR, number of samples per once Euclidean distance computing. Default: `10000`.

+ `--col-batch-size` parameter to calculate PR, number of samples per once Euclidean distance computing. Default: `10000`.

+ `--dims` as pytorch-fid package, there are four available positions to extract features from Inception module. Available options: `64, 192, 768, 2048`. Default: `2048`.
    + 64: first max pooling features.
    + 192: second max pooling features.
    + 768: pre-aux classifier features.
    + 2048: final average pooling features. 

+ `--batch-size` Batch size. Default: 50.

+ `--num-workers` The number of workers use for dataloder. Default: 8.

+ `--not-save-cache` Store true parameter. Call it if you don't want to save dataset statistic as cache.

+ `--verbose` Store true parameter. Call it if you want to see the process bar while computing statistics information from the image folder.



