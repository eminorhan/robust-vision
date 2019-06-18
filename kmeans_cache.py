"""Do k-means on cache
"""
import argparse
import numpy as np
from keras.utils import to_categorical
from sklearn.cluster import MiniBatchKMeans

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

short_vals = np.argmax(mem_vals, axis=1)

for i in range(1000):
    mem_keys_class = mem_keys[short_vals==i, :]

    class_size = mem_keys_class.shape[0]
    n_clusters = class_size // args.reduce_factor + 1
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=128)
    kmeans.fit(mem_keys_class)

    if i == 0:
        mem_keys_running = kmeans.cluster_centers_
        mem_vals_running = to_categorical(i * np.ones(n_clusters), num_classes=1000)
    else:
        mem_keys_running = np.concatenate((mem_keys_running, kmeans.cluster_centers_), axis=0)
        mem_vals_running = np.concatenate((mem_vals_running, to_categorical(i * np.ones(n_clusters), num_classes=1000)),
                                          axis=0)

    print('Completed k-means on class %i'%i)

print('Successfully completed online k-means')
print('Key matrix shape:', mem_keys_running.shape)
print('Value matrix shape:', mem_vals_running.shape)

# save results
np.savez(args.mem_save_dir + 'ResNet50_cache_all_kmeans%i_layer%i.npz'%(args.reduce_factor, mem_layer),
         mem_keys=mem_keys_running, mem_vals=mem_vals_running)