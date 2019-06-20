"""Evaluate cache models with Shape-ResNet-50 backbone on clean images
"""
import argparse
from collections import OrderedDict
import numpy as np
import torch
import torchvision.models
from keras.applications.resnet50 import preprocess_input
from scipy.io import loadmat
from utils import LastLayerModel, load_model


parser = argparse.ArgumentParser(description='Evaluate cache models with Shape-ResNet-50 backbone on clean images')
parser.add_argument('--val_data_dir', type=str, help='Validation data directory')
parser.add_argument('--model_save_dir', type=str,
                    help='directory where the pre-trained Shape-ResNet models are stored')
parser.add_argument('--mem_save_dir', type=str, help='directory where the caches are stored')
parser.add_argument('--layer', type=str, default='activation_46', choices=['fc1000', 'avg_pool', 'activation_46'],
                    help='which layer to use for embedding')
parser.add_argument('--theta', type=float, default=60.0, help='theta hyper-parameter (inverse temperature)')

args = parser.parse_args()

val_data_dir = args.val_data_dir
mem_save_dir = args.mem_save_dir
model_save_dir = args.model_save_dir
theta = args.theta

if args.layer == 'fc1000':
    which_layer = 176
elif args.layer == 'avg_pool':
    which_layer = 175
elif args.layer == 'activation_46':
    which_layer = 164
else:
    raise ValueError('Layer not supported for caching.')

print('Theta:', theta)
print('Layer:', which_layer)

model_name = 'resnet50_sin_in_in'  # change to different model as desired
model = load_model(model_name, which_layer, model_save_dir)
print("Model loading completed.")

num_batches = 10
num_imgs_per_batch = 5000

num_inner_batches = 8
num_imgs_per_inner_batch = 625

y_val_all = np.loadtxt('ILSVRC2012_validation_ground_truth.txt', usecols=1)  # class labels for val data

# Load cache
data = np.load(mem_save_dir + 'ResNet50_sin_in_in_layer%i.npz'%which_layer)
mem_keys, mem_vals = data['mem_keys'], data['mem_vals']
mem_keys = mem_keys / np.linalg.norm(mem_keys, axis=1, keepdims=True)
print('Successfully loaded the cache and the projection matrix.')

model.eval()

with torch.no_grad():

    mem_accs = np.zeros(num_batches)

    for val_batch in range(num_batches):
        # Data for the current batch
        x_val = loadmat(val_data_dir + 'val_batch_%i'%(val_batch + 1))['all_imgs']
        x_val = preprocess_input(x_val.transpose((0, 3, 1, 2)), data_format='channels_first', mode='torch')
        y_val = y_val_all[(val_batch * num_imgs_per_batch):((val_batch + 1) * num_imgs_per_batch)]

        mem_acc_inner = np.zeros(num_inner_batches)

        for inner_batch in range(num_inner_batches):

            x_val_inner = x_val[(inner_batch * num_imgs_per_inner_batch):((inner_batch + 1) *
                                                                          num_imgs_per_inner_batch), ...]
            x_val_inner = torch.from_numpy(np.float32(x_val_inner)).cuda()
            y_val_inner = y_val[(inner_batch * num_imgs_per_inner_batch):((inner_batch + 1) *
                                                                          num_imgs_per_inner_batch)]

            preds = torch.squeeze(model(x_val_inner))
            preds = preds.cpu().numpy()

            # Normalize query
            query = preds / np.linalg.norm(preds, axis=1, keepdims=True)

            similarities = np.exp(theta * np.dot(query, mem_keys.T))
            p_mem = np.matmul(similarities, mem_vals)
            p_mem = p_mem / np.sum(p_mem, axis=1, keepdims=True)
            pred_mem = np.argmax(p_mem, axis=1)

            mem_acc_inner[inner_batch] = np.mean(pred_mem==y_val_inner)

        mem_accs[val_batch] = np.mean(mem_acc_inner)

        print('Batch:', val_batch, 'Mem. Batch acc:', mem_accs[val_batch])

print('Overall mean accuracy (cache):', np.mean(mem_accs))

np.savez('cache_clean_accuracy_layer%i_theta%2.f.npz'%(which_layer, theta),
         mem_acc=np.mean(mem_accs), layer=which_layer, theta=theta)