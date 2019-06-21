"""Evaluate cache models on ImageNet-C
"""
import argparse
import numpy as np
from scipy.io import loadmat
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input

parser = argparse.ArgumentParser(description='Evaluate cache models on clean images')
parser.add_argument('--val_data_dir', type=str, help='directory where pre-processed ImageNet-C data is stored')
parser.add_argument('--mem_save_dir', type=str, help='directory where the caches are stored')
parser.add_argument('--layer', type=str, default='activation_46',
                    choices=['fc1000', 'avg_pool', 'activation_46', 'activation_43'],
                    help='which layer to use for embedding')
parser.add_argument('--theta', type=float, default=50.0, help='theta hyper-parameter (inverse temperature)')
parser.add_argument('--corruption', type=str, choices=['brightness', 'contrast', 'defocus_blur', 'elastic_transform',
                                                       'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur',
                                                       'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate',
                                                       'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise',
                                                       'zoom_blur'], help='corruption type')
parser.add_argument('--severity', type=str, choices = ['1', '2', '3', '4', '5'], help='severity level')

args = parser.parse_args()

val_data_dir = args.val_data_dir + args.corruption + '/' + args.severity + '/'
mem_save_dir = args.mem_save_dir
theta = args.theta

if args.layer == 'fc1000':
    mem_layer = 176
elif args.layer == 'avg_pool':
    mem_layer = 175
elif args.layer == 'activation_46':
    mem_layer = 164
elif args.layer == 'activation_43':
    mem_layer = 154
else:
    raise ValueError('Layer not supported for caching.')

# Define model
model = ResNet50(weights='imagenet')

if mem_layer == 164 or mem_layer == 154:
    output = GlobalAveragePooling2D()(model.layers[mem_layer].output)
else:
    output = model.layers[mem_layer].output

print('Cache layer:', model.layers[mem_layer].name)

# Define memory
mem = Model(inputs=model.input, outputs=output)

# Load cache
data = np.load(mem_save_dir + 'ResNet50_cache_all_layer%i.npz' % mem_layer)
mem_keys, mem_vals = data['mem_keys'], data['mem_vals']
key = mem_keys / np.linalg.norm(mem_keys, axis=1, keepdims=True)

print('Successfully loaded the cache.')

num_batches = 1000
num_imgs_per_batch = 50
val_accs_mem = np.zeros(num_batches)

# Pass imgs thru memory
for val_batch in range(num_batches):
    x_val = loadmat(val_data_dir + 'class_%i.mat' % val_batch)['all_imgs']
    x_val = preprocess_input(x_val)
    y_val = loadmat(val_data_dir + 'class_%i.mat' % val_batch)['all_labels']

    test_mem = mem.predict(x_val)

    # Normalize query
    query = test_mem / np.linalg.norm(test_mem, axis=1, keepdims=True)

    similarities = np.exp(theta * np.dot(query, key.T))
    p_mem = np.matmul(similarities, mem_vals)
    p_mem = p_mem / np.sum(p_mem, axis=1, keepdims=True)

    pred_mem = np.argmax(p_mem, axis=1)
    test_acc = np.mean(pred_mem==y_val)

    print('Mem. accuracy:', test_acc)
    val_accs_mem[val_batch] = test_acc

print('Mean mem. accuracy:', np.mean(val_accs_mem))

np.savez('cache_imagenetc_accuracy_' + args.corruption + args.severity + '_layer%i_theta%2.f.npz' % (mem_layer, theta),
         mem_acc=np.mean(val_accs_mem), layer=mem_layer, theta=theta)
