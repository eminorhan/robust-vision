"""Pre-process ImageNet validation images
"""
import os
import argparse
import numpy as np
import scipy.io as sio
import cv2

parser = argparse.ArgumentParser(description='Pre-process ImageNet validation data')
parser.add_argument('--source_dir', type=str, help='directory where ImageNet validation data is stored')
parser.add_argument('--target_dir', type=str, help='directory where the pre-processed data will be stored')

args = parser.parse_args()

num_batches = 10
num_imgs_per_batch = 5000

file_list = os.listdir(args.source_dir)
file_list.sort()

labels = np.loadtxt('ILSVRC2012_validation_ground_truth.txt', usecols=1)

for i in range(num_batches):
    all_imgs = []
    for file_indx in file_list[(i*num_imgs_per_batch):((i+1)*num_imgs_per_batch)]:
        file_name = args.source_dir + file_indx

        img = cv2.imread(file_name)

        # Resize
        height, width, _ = img.shape
        new_height = height * 256 // min(img.shape[:2])
        new_width = width * 256 // min(img.shape[:2])
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Crop
        height, width, _ = img.shape
        startx = width // 2 - (224 // 2)
        starty = height // 2 - (224 // 2)
        img = img[starty:starty + 224, startx:startx + 224]
        assert img.shape[0] == 224 and img.shape[1] == 224, (img.shape, height, width)

        x = img[:, :, ::-1]  # OpenCV assumes BGR. This converts it back to RGB.
        x = np.expand_dims(x, axis=0)

        all_imgs.append(x)

    all_imgs = np.squeeze(np.asarray(all_imgs))
    all_labels = labels[(i*num_imgs_per_batch):((i+1)*num_imgs_per_batch)]

    print(all_imgs.shape)
    print(all_labels.shape)

    sio.savemat(args.target_dir + 'val_batch_%i.mat'%(i+1), {'all_imgs': all_imgs, 'all_labels': all_labels})