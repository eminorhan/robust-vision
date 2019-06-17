"""Pre-process ImageNet training data
"""
import os
import argparse
import numpy as np
import scipy.io as sio
import cv2

parser = argparse.ArgumentParser(description='Pre-process ImageNet training data')
parser.add_argument('--source_dir', type=str, help='directory where ImageNet training data is stored')
parser.add_argument('--target_dir', type=str, help='directory where the pre-processed data will be stored')

args = parser.parse_args()

dir_list = os.listdir(args.source_dir)
dir_list.sort()
print(dir_list)

dir_itr = 0

for dir_indx in dir_list:
    dir_name = args.source_dir + dir_indx
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

        x = img[:, :, ::-1]  # OpenCV assumes BGR. This converts it back to RGB.
        x = np.expand_dims(x, axis=0)

        all_imgs.append(x)

    all_imgs = np.squeeze(np.asarray(all_imgs))
    all_labels = dir_itr * np.ones(all_imgs.shape[0])
    sio.savemat(args.target_dir + 'class_%i.mat' % dir_itr, {'all_imgs': all_imgs, 'all_labels': all_labels})
    print('Directory %i of %i'%(dir_itr, len(dir_list)))
    dir_itr = dir_itr + 1