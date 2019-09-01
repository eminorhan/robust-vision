"""Make and store the cache (with a Shape-ResNet-50 backbone)
"""
import argparse
from collections import OrderedDict
import numpy as np
import torch
import torchvision.models
from keras.utils import to_categorical
from keras.applications.resnet50 import preprocess_input
from scipy.io import loadmat
from utils import LastLayerModel


parser = argparse.ArgumentParser(description='Make and store caches with a Shape-ResNet-50 backbone')
parser.add_argument('--model_save_dir', type=str,
                    help='directory where the pre-trained Shape-ResNet model is stored')
parser.add_argument('--train_data_dir', type=str,
                    help='directory where pre-processed ImageNet training data is stored')
parser.add_argument('--mem_save_dir', type=str, help='directory where the caches will be stored')
parser.add_argument('--layer', type=str, default='activation_46', choices=['fc1000', 'avg_pool', 'activation_46'],
                    help='which layer to use for embedding')

args = parser.parse_args()
model_save_dir = arge.model_save_dir
train_data_dir = args.train_data_dir
mem_save_dir = args.mem_save_dir

n_classes = 1000

if args.layer == 'fc1000':
    which_layer = 176
elif args.layer == 'avg_pool':
    which_layer = 175
elif args.layer == 'activation_46':
    which_layer = 164
else:
    raise ValueError('Layer not supported for caching.')

# Specify model
model_name = 'resnet50_sin_in_in'

model = torchvision.models.resnet50(pretrained=False)
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load(model_save_dir + model_name + '.pth.tar'))

if which_layer==176:
    x = list(model.module.children())
elif which_layer==175:
    x = list(model.module.children())[:-1]
elif which_layer==164:
    x = list(model.module.children())[:-3]
    y = list(model.module.children())[-3]
    z = list(y.children())[:-1]
    x.append(torch.nn.Sequential(*z))
    x.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))

if which_layer==176:
    new_model = LastLayerModel(model)
else:
    new_model = torch.nn.Sequential(*x)

print(new_model)

all_mem_keys = []
all_mem_vals = []

new_model.eval()

with torch.no_grad():

    for cl_idx in range(n_classes):
        y_train = loadmat(train_data_dir + 'class_%i.mat'%cl_idx)['all_labels']
        y_train = y_train[0, :]

        # Memory values
        mem_vals = y_train
        all_mem_vals.append(mem_vals)

        # preprocess for pytorch
        data_x = loadmat(train_data_dir + 'class_%i.mat'%cl_idx)['all_imgs']

        # First batch
        x_train_1 = data_x[:625, :, :, :]
        x_train_1 = preprocess_input(x_train_1.transpose((0, 3, 1, 2)), data_format='channels_first', mode='torch')
        x_train_1 = torch.from_numpy(x_train_1).cuda()

        # Memory keys
        mem_keys = torch.squeeze(new_model(x_train_1))
        mem_keys = mem_keys.cpu().numpy()
        all_mem_keys.append(mem_keys)

        # Second batch
        x_train_2 = data_x[625:, :, :, :]
        x_train_2 = preprocess_input(x_train_2.transpose((0, 3, 1, 2)), data_format='channels_first', mode='torch')
        x_train_2 = torch.from_numpy(x_train_2).cuda()

        # Memory keys
        mem_keys = torch.squeeze(new_model(x_train_2))
        mem_keys = mem_keys.cpu().numpy()

        # if which_layer==176:
        #     print('Softmax test 0:', np.min(mem_keys[0, :]), np.max(mem_keys[0, :]), np.sum(mem_keys[0, :]))
        #     print('Softmax test 1:', np.min(mem_keys[1, :]), np.max(mem_keys[1, :]), np.sum(mem_keys[1, :]))
        #     print('Softmax test 2:', np.min(mem_keys[2, :]), np.max(mem_keys[2, :]), np.sum(mem_keys[2, :]))

        all_mem_keys.append(mem_keys)

        if cl_idx % 10==0:
            print('Iter %i of %i' % (cl_idx, n_classes))

    mem_keys = np.concatenate(all_mem_keys, axis=0)
    mem_vals = np.concatenate(all_mem_vals, axis=0)
    mem_vals = to_categorical(mem_vals, n_classes)

    print('Key matrix shape:', mem_keys.shape)
    print('Value matrix shape:', mem_vals.shape)

    np.savez(mem_save_dir + 'ResNet50_sin_in_in_layer%i.npz' % which_layer, mem_keys=mem_keys, mem_vals=mem_vals)
    print('Successfully saved the cache.')
