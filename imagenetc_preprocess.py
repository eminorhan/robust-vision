"""Pre-process ImageNet-C
"""
import os
import argparse
import numpy as np
import scipy.io as sio
import cv2

parser = argparse.ArgumentParser(description='Pre-process ImageNet validation data')
parser.add_argument('--source_dir', type=str, help='directory where raw ImageNet-C data is stored')
parser.add_argument('--target_dir', type=str, help='directory where the pre-processed data will be stored')
parser.add_argument('--corruption', type=str, choices=['brightness', 'contrast', 'defocus_blur', 'elastic_transform',
                                                       'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur',
                                                       'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate',
                                                       'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise',
                                                       'zoom_blur'], help='corruption type')
parser.add_argument('--severity', type=str, choices = ['1', '2', '3', '4', '5'], help='severity level')

args = parser.parse_args()

base_dir = args.source_dir + args.corruption + '/' + args.severity + '/'
proc_val_dir = args.target_dir + args.corruption + '/' + args.severity + '/'

dir_list = os.listdir(base_dir)
dir_list.sort()
print(dir_list)

dir_itr = 0

for dir_indx in dir_list:
    dir_name = base_dir + dir_indx
    file_list = os.listdir(dir_name)
    file_list.sort()

    all_imgs = []
    for file_indx in file_list:
        file_name = dir_name + '/' + file_indx

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

        x = img[:, :, ::-1]
        x = np.expand_dims(x, axis=0)

        all_imgs.append(x)

    all_imgs = np.squeeze(np.asarray(all_imgs))
    all_labels = dir_itr * np.ones(all_imgs.shape[0])
    sio.savemat(proc_val_dir + 'class_%i.mat' % dir_itr, {'all_imgs': all_imgs, 'all_labels': all_labels})
    print('Directory %i of %i'%(dir_itr,len(dir_list)))
    dir_itr = dir_itr + 1