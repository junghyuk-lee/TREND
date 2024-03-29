import torchvision.transforms as transforms
from trend import extract_embeddings, estimate_params, compute_jsd

transform = transforms.Compose(
    [transforms.Resize((224,224)),
    transforms.ToTensor()])

# estimated parameters from real dataset: ImageNet
params_r = './params_imagenet.npy'

# test dataset (e.g., images generated by BigGAN)
dir_sub = './data/biggan'

# get Inception features
activations = extract_embeddings(dir_sub, n_images=50000, transform=transform)

# estimate parameters of truncated generalized Gaussian distribution (mu, sigma, beta)
params = estimate_params(activations)

# compute jsd between real and generated distributions for trend measure
trend = compute_jsd(params_r,params)

print(trend)