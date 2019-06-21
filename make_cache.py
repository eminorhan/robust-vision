"""Make and store the cache (with a ResNet-50 backbone).
"""
import argparse
import numpy as np
from scipy.io import loadmat
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50, preprocess_input

parser = argparse.ArgumentParser(description='Make and store the caches with a ResNet-50 backbone')
parser.add_argument('--train_data_dir', type=str, help='directory where pre-processed ImageNet training data is stored')
parser.add_argument('--mem_save_dir', type=str, help='directory where the caches will be stored')
parser.add_argument('--layer', type=str, default='activation_46',
                    choices=['fc1000', 'avg_pool', 'activation_46', 'activation_43'],
                    help='which layer to use for embedding')

args = parser.parse_args()

train_data_dir = args.train_data_dir
mem_save_dir = args.mem_save_dir
n_classes = 1000

if args.layer == 'fc1000':
    mem_layer = 176
elif args.layer == 'avg_pool':
    mem_layer = 175
elif args.layer == 'activation_46':
    mem_layer = 164
elif args.layer == 'activation_43':
    mem_layer = 154
else:
    raise ValueError('Layer not allowed for caching.')

# Define model
model = ResNet50(weights='imagenet')

if mem_layer == 164 or mem_layer == 154:
    output = GlobalAveragePooling2D()(model.layers[mem_layer].output)
else:
    output = model.layers[mem_layer].output

print('Cache layer:', model.layers[mem_layer].name)

# Define memory
mem = Model(inputs=model.input, outputs=output)

all_mem_keys = []
all_mem_vals = []

for cl_idx in range(n_classes):
    x_train = loadmat(train_data_dir + 'class_%i.mat' % cl_idx)['all_imgs']
    y_train = loadmat(train_data_dir + 'class_%i.mat' % cl_idx)['all_labels']

    x_train = preprocess_input(x_train)
    y_train = y_train[0, :]

    # Memory keys
    mem_keys = mem.predict(x_train)

    # Memory values
    mem_vals = y_train

    all_mem_keys.append(mem_keys)
    all_mem_vals.append(mem_vals)

    print('Iter %i of %i' % (cl_idx, n_classes))

mem_keys = np.concatenate(all_mem_keys, axis=0)
mem_vals = np.concatenate(all_mem_vals, axis=0)
mem_vals = to_categorical(mem_vals, n_classes)

print('Key matrix shape:', mem_keys.shape)
print('Value matrix shape:', mem_vals.shape)

np.savez(mem_save_dir + 'ResNet50_cache_all_layer%i.npz' % mem_layer, mem_keys=mem_keys, mem_vals=mem_vals)

print('Successfully saved the cache.')
