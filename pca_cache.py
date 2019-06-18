"""Do online PCA on cache
"""
import argparse
from sklearn.decomposition import IncrementalPCA
import numpy as np

parser = argparse.ArgumentParser(description='Do online PCA on cache')
parser.add_argument('--mem_save_dir', type=str, help='directory where the caches are stored')
parser.add_argument('--layer', type=str, default='activation_46',
                    choices=['fc1000', 'avg_pool', 'activation_46', 'activation_43'],
                    help='which layer to use for embedding')
parser.add_argument('--reduce_factor', type=int, default=8, help='dimensionality reduction factor')

args = parser.parse_args()

if args.layer == 'fc1000':
    mem_layer = 176
elif args.layer == 'avg_pool':
    mem_layer = 175
elif args.layer == 'activation_46':
    mem_layer = 164
elif args.layer == 'activation_43':
    mem_layer = 154
else:
    raise ValueError('Layer not available for caching.')

data = np.load(args.mem_save_dir + 'ResNet50_cache_all_layer%i.npz'%mem_layer)
mem_keys, mem_vals = data['mem_keys'], data['mem_vals']
print('Successfully loaded the cache.')

n_components = mem_keys.shape[1] // args.reduce_factor

# do incremental PCA
ipca = IncrementalPCA(n_components=n_components)
mem_keys_ipca = ipca.fit_transform(mem_keys)
W_proj = ipca.components_.T

print('Successfully completed online PCA. Variance explained by top %i PCs is'%n_components,
      np.sum(ipca.explained_variance_ratio_))

# save results
np.savez(args.mem_save_dir + 'ResNet50_cache_all_pca%i_layer%i.npz'%(args.reduce_factor, mem_layer),
         mem_keys=mem_keys_ipca, mem_vals=mem_vals, W_proj=W_proj)