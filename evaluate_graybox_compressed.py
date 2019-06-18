"""Evaluate compressed cache models on gray-box adversarial images
"""
import argparse
import numpy as np
from scipy.io import loadmat
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input

parser = argparse.ArgumentParser(description='Evaluate compressed cache models on gray-box adversarial images')
parser.add_argument('--graybox_data_dir', type=str, help='directory where graybox adversarial images are stored')
parser.add_argument('--mem_save_dir', type=str, help='directory where the caches are stored')
parser.add_argument('--layer', type=str, default='activation_46',
                    choices=['fc1000', 'avg_pool', 'activation_46', 'activation_43'],
                    help='which layer to use for embedding')
parser.add_argument('--theta', type=float, default=50.0, help='theta hyper-parameter (inverse temperature)')
parser.add_argument('--epsilon', type=float, default=0.06, help='Perturbation size (normalized l2-norm)')
parser.add_argument('--reduce_method', type=str, default='kmeans', choices=['pca', 'kmeans'],
                    help='compression method (pca or k-means)')
parser.add_argument('--reduce_factor', type=int, default=8, help='compression factor')

args = parser.parse_args()

graybox_data_dir = args.graybox_data_dir
mem_save_dir = args.mem_save_dir
theta = args.theta
epsilon = args.epsilon

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

eps_str = 'eps%.2f'%epsilon

print('Theta:', theta)
print('Eps:', epsilon)

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
data = np.load(mem_save_dir + 'ResNet50_cache_all_' + args.reduce_method + str(args.reduce_factor) +
               '_layer%i.npz'%mem_layer)

mem_keys, mem_vals = data['mem_keys'], data['mem_vals']
key = mem_keys / np.linalg.norm(mem_keys, axis=1, keepdims=True)

if args.reduce_method == 'pca':
    W_proj = data['W_proj']

print('Successfully loaded the cache.')

num_batches = 10
num_imgs_per_batch = 5000
adv_accs_mem = np.zeros(num_batches)
adv_accs_nomem = np.zeros(num_batches)

val_data_dir = graybox_data_dir + 'graybox_adversarial/l2/' + eps_str + '/'

# Pass adversarial imgs thru memory
for val_batch in range(num_batches):
    batch_data = np.load(val_data_dir + 'ResNet50_graybox_l2_advs_%i.npz'%(val_batch + 1))
    x_val = batch_data['adv_images']
    x_val = preprocess_input(x_val[..., ::-1])

    y_val = batch_data['clean_labels']
    nomem_preds = batch_data['adv_pred_labels']  # model.predict(x_val)

    test_mem = mem.predict(x_val)

    if args.reduce_method == 'pca':
        test_mem = np.dot(test_mem, W_proj)

    # Normalize query
    query = test_mem / np.linalg.norm(test_mem, axis=1, keepdims=True)

    similarities = np.exp(theta * np.dot(query, key.T))
    p_mem = np.matmul(similarities, mem_vals)
    p_mem = p_mem / np.sum(p_mem, axis=1, keepdims=True)

    pred_mem = np.argmax(p_mem, axis=1)
    adv_acc = np.mean(pred_mem==y_val)

    no_mem_acc = np.mean(nomem_preds==y_val)

    print('Mem. accuracy:', adv_acc)
    adv_accs_mem[val_batch] = adv_acc

    print('No mem. accuracy:', no_mem_acc)
    adv_accs_nomem[val_batch] = no_mem_acc

print('Mean mem. accuracy:', np.mean(adv_accs_mem))
print('Mean no mem. accuracy:', np.mean(adv_accs_nomem))

np.savez('compressed_cache_graybox_accuracy_' + args.reduce_method + str(args.reduce_factor) + '_' + eps_str +
         '_layer%i_theta%2.f.npz'%(mem_layer, theta),
         mem_acc = np.mean(adv_accs_mem),
         no_mem_acc = np.mean(adv_accs_nomem),
         layer = mem_layer,
         theta = theta,
         epsilon = epsilon
         )
