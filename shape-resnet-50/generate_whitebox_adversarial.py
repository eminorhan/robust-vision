"""Run white-box adversarial attacks against cache models
"""
import argparse
from collections import OrderedDict
import torch
import torchvision.models
from foolbox.models import PyTorchModel
from foolbox.attacks import RandomStartProjectedGradientDescentAttack
from foolbox.criteria import TargetClass
from utils import Ltwo
import numpy as np
from scipy.io import loadmat
from utils import LastLayerModel, CacheModel, load_model


parser = argparse.ArgumentParser(description='Run white-box attacks against cache models')
parser.add_argument('--base_dir', type=str, help='Base directory')
parser.add_argument('--model_save_dir', type=str,
                    help='directory where the pre-trained Shape-ResNet models are stored')
parser.add_argument('--mem_save_dir', type=str, help='directory where the caches are stored')
parser.add_argument('--layer', type=str, default='activation_46',
                    choices=['fc1000', 'avg_pool', 'activation_46'],
                    help='which layer to use for embedding')
parser.add_argument('--theta', type=float, default=50.0, help='theta hyper-parameter (inverse temperature)')
parser.add_argument('--epsilon', type=float, default=0.06, help='Perturbation size (normalized l2-norm)')

args = parser.parse_args()

val_data_dir = args.base_dir + 'processed_val/'
mem_save_dir = args.mem_save_dir
model_save_dir = args.model_save_dir

epsilon = args.epsilon
theta = args.theta

if args.layer == 'fc1000':
    which_layer = 176
elif args.layer == 'avg_pool':
    which_layer = 175
elif args.layer == 'activation_46':
    which_layer = 164
else:
    raise ValueError('Layer not supported for caching.')

n_classes = 1000

# Load cache
data = np.load(mem_save_dir + 'ResNet50_sin_in_in_layer%i.npz'%which_layer)

if which_layer==176:
    mem_keys, mem_vals = data['mem_keys'], data['mem_vals']
else:
    mem_keys1, mem_vals1 = data['mem_keys'][::2, :], data['mem_vals'][::2, :]  # evens
    mem_keys2, mem_vals2 = data['mem_keys'][1::2, :], data['mem_vals'][1::2, :]  # odds
    mem_keys = np.concatenate((mem_keys1, mem_keys2[::2, :]), axis=0)
    mem_vals = np.concatenate((mem_vals1, mem_vals2[::2, :]), axis=0)

mem_keys = mem_keys / np.linalg.norm(mem_keys, axis=1, keepdims=True)

print(mem_keys.shape)
print(mem_vals.shape)
print('Successfully loaded the cache and the projection matrix, if it exists.')

# Specify model
model_name = 'resnet50_sin_in_in'
backbone = load_model(model_name, which_layer, model_save_dir)  # change to different model as desired
model = CacheModel(backbone, mem_keys.T, mem_vals, theta)

print("Model loading completed.")

mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
preprocessing = (mean, std)

fmodel = PyTorchModel(model.eval(), bounds=(0, 1), num_classes=n_classes, preprocessing=preprocessing)
print('Foolbox model successfully loaded.')

x_val = loadmat(val_data_dir + 'val_batch_%i' %1)['all_imgs']
x_val = x_val.transpose((0, 3, 1, 2))
x_val = np.float32(x_val / 255.0)

y_val = np.loadtxt('ILSVRC2012_validation_ground_truth.txt', usecols=1)

num_imgs_per_batch = 1000  # number of images to run attacks against

adv_images = np.zeros((num_imgs_per_batch, 3, 224, 224))
clean_labels = np.zeros(num_imgs_per_batch)
target_labels = np.zeros(num_imgs_per_batch)
clean_pred_labels = np.zeros(num_imgs_per_batch)
adv_pred_labels = np.zeros(num_imgs_per_batch)

for img_i in range(num_imgs_per_batch):
    image, clean_label = x_val[img_i, ...], y_val[img_i]

    target_label = np.random.choice(np.setdiff1d(np.arange(n_classes), clean_label))
    attack = RandomStartProjectedGradientDescentAttack(model=fmodel, criterion=TargetClass(target_label),
                                                       distance=Ltwo)
    adversarial = attack(image, clean_label, binary_search=False, epsilon=epsilon, stepsize=2./255,
                         iterations=10, random_start=True)

    if np.any(adversarial==None):
        adversarial=image
        target_label=clean_label
        adv_pred_labels[img_i] = np.argmax(fmodel.predictions(adversarial))
        # print('No adversarial!')
    else:
        adv_pred_labels[img_i] = np.argmax(fmodel.predictions(adversarial))

    adv_images[img_i, ...] = adversarial
    clean_labels[img_i] = clean_label
    target_labels[img_i] = target_label
    clean_pred_labels[img_i] = np.argmax(fmodel.predictions(image))

    print('Iter, Clean, Clean_pred, Adv, Adv_pred: ', img_i, clean_label, clean_pred_labels[img_i], target_label,
          adv_pred_labels[img_i])

mem_acc = np.mean(clean_labels==adv_pred_labels)
print('Mean adversarial accuracy (cache):', mem_acc)

# save adversarial images and target labels
np.savez('cache_whitebox_accuracy_' + 'eps%.2f'%epsilon + '_layer%i_theta%2.f.npz'%(which_layer, theta),
         mem_acc=mem_acc,
         clean_labels=clean_labels,
         target_labels=target_labels,
         clean_pred_labels=clean_pred_labels,
         adv_pred_labels=adv_pred_labels
         )