"""Evaluate cache models with Shape-ResNet-50 backbone on gray-box adversarial images
"""
import argparse
from collections import OrderedDict
import numpy as np
import torch
import torchvision.models
from keras.applications.resnet50 import preprocess_input
from utils import LastLayerModel, load_model


parser = argparse.ArgumentParser(description='Evaluate cache models with Shape-ResNet-50 backbone on graybox images')
parser.add_argument('--graybox_data_dir', type=str, help='directory where graybox adversarial images are stored')
parser.add_argument('--model_save_dir', type=str,
                    help='directory where the pre-trained Shape-ResNet models are stored')
parser.add_argument('--mem_save_dir', type=str, help='directory where the caches are stored')
parser.add_argument('--layer', type=str, default='activation_46', choices=['fc1000', 'avg_pool', 'activation_46'],
                    help='which layer to use for embedding')
parser.add_argument('--theta', type=float, default=60.0, help='theta hyper-parameter (inverse temperature)')
parser.add_argument('--epsilon', type=float, default=0.06, help='Perturbation size (normalized l2-norm)')

args = parser.parse_args()

graybox_data_dir = args.graybox_data_dir
mem_save_dir = args.mem_save_dir
model_save_dir = args.model_save_dir
theta = args.theta
epsilon = args.epsilon

if args.layer == 'fc1000':
    which_layer = 176
elif args.layer == 'avg_pool':
    which_layer = 175
elif args.layer == 'activation_46':
    which_layer = 164
else:
    raise ValueError('Layer not supported for caching.')

eps_str = 'eps%.2f' % epsilon

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
data = np.load(mem_save_dir + 'ResNet50_sin_in_in_layer%i.npz' % which_layer)
mem_keys, mem_vals = data['mem_keys'], data['mem_vals']
mem_keys = mem_keys / np.linalg.norm(mem_keys, axis=1, keepdims=True)
print('Successfully loaded the cache.')

model.eval()

channel_means = [0.485, 0.456, 0.406]
channel_stds = [0.229, 0.224, 0.225]

val_data_dir = graybox_data_dir + 'graybox_adversarial/l2/' + eps_str + '/'

with torch.no_grad():

    mem_accs = np.zeros(num_batches)
    no_mem_accs = np.zeros(num_batches)

    for val_batch in range(num_batches):
        # Data for the current batch
        batch_data = np.load(val_data_dir + 'ResNet50_graybox_l2_advs_%i.npz' % (val_batch + 1))
        x_val = batch_data['adv_images']

        x_val[:, 0, :, :] -= channel_means[0]
        x_val[:, 1, :, :] -= channel_means[1]
        x_val[:, 2, :, :] -= channel_means[2]

        x_val[:, 0, :, :] /= channel_stds[0]
        x_val[:, 1, :, :] /= channel_stds[1]
        x_val[:, 2, :, :] /= channel_stds[2]

        y_val = y_val_all[(val_batch * num_imgs_per_batch):((val_batch + 1) * num_imgs_per_batch)]

        nomem_preds = batch_data['adv_pred_labels']

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
        no_mem_accs[val_batch] = np.mean(nomem_preds==y_val)

        print('Batch:', val_batch, 'Mem. Batch acc:', mem_accs[val_batch],
              'Nomem. Batch acc:', no_mem_accs[val_batch])

print('Overall mean accuracy (cache):', np.mean(mem_accs))
print('Overall mean accuracy (no cache):', np.mean(no_mem_accs))

np.savez('cache_graybox_accuracy_' + eps_str + '_layer%i_theta%2.f.npz' % (which_layer, theta),
         mem_acc=np.mean(mem_accs),
         no_mem_acc=np.mean(no_mem_accs),
         layer=which_layer,
         theta=theta
         )
