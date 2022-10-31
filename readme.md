# Code for ECCV submission ID - 3604
Code for ECCV submission ID - 3604 "TREND: Truncated Generalized Normal Density Estimation of Inception Embeddings for Accurate GAN Evaluation"

## Requirements
- Python 3.7
- PyTorch
- pytorch_fid
<!-- 
pip install foobar
``` -->

## Data Preparation
The images should be arranged in this way:

    ./data/[test gan]/*/xxx.png
    ./data/[test gan]/*/xxy.png
    ./data/[test gan]/*/xxz.png
    * can be any folder in [test gan]

## Usage
Run `main.py`
```python
import torchvision.transforms as transforms
from trend import extract_embeddings, estimate_params, compute_jsd

transform = transforms.Compose(
    [transforms.Resize((224,224)),
    transforms.ToTensor()])

# estimated parameters from real dataset: ImageNet
params_r = './params_imagenet.npy'

# dir to the test dataset (e.g., images generated by BigGAN)
dir_sub = './data/biggan'

# get Inception features
activations = extract_embeddings(dir_sub, n_images=50000, transform=transform)

# estimate parameters of truncated generalized normal distribution (mu, sigma, beta)
params = estimate_params(activations)

# compute jsd between real and generated distributions for trend measure
trend = compute_jsd(params_r,params)

print(trend)
```