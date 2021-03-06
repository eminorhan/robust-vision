"""Generate gray-box adversarial images
"""
import argparse
from collections import OrderedDict
import numpy as np
import torch
import torchvision.models
from foolbox.models import PyTorchModel
from foolbox.attacks import RandomStartProjectedGradientDescentAttack
from foolbox.criteria import TargetClass
from scipy.io import loadmat
from utils import Ltwo


def load_model(model_name, model_save_dir):
    "Loads one of the pretrained SIN models."

    model = torchvision.models.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(model_save_dir + model_name + '.pth.tar'))
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate gray-box adversarial images')
    parser.add_argument('--base_dir', type=str, help='Base directory')
    parser.add_argument('--model_save_dir', type=str,
                        help='directory where the pre-trained Shape-ResNet models are stored')
    parser.add_argument('--epsilon', type=float, default=0.06, help='Perturbation size (normalized l2-norm)')

    args = parser.parse_args()

    val_data_dir = args.base_dir + 'processed_val/'
    adv_save_dir = args.base_dir + 'graybox_adversarial/l2/'
    model_save_dir = args.model_save_dir
    epsilon = args.epsilon

    n_classes = 1000

    # Specify model
    model_name = 'resnet50_sin_in_in'  # change to different model as desired
    model = load_model(model_name, model_save_dir)  
    print("Model loading completed.")

    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    preprocessing = (mean, std)

    fmodel = PyTorchModel(model.eval(), bounds=(0, 1), num_classes=n_classes, preprocessing=preprocessing)

    num_batches = 10
    num_imgs_per_batch = 5000
    val_accs_mem = np.zeros(num_batches)

    val_ground_truth = np.loadtxt('ILSVRC2012_validation_ground_truth.txt', usecols=1)

    for val_batch in range(num_batches):
        x_val = loadmat(val_data_dir + 'val_batch_%i'%(val_batch + 1))['all_imgs']
        x_val = x_val.transpose((0, 3, 1, 2))
        x_val = np.float32(x_val / 255.0)

        y_val = val_ground_truth[(val_batch * num_imgs_per_batch):((val_batch + 1) * num_imgs_per_batch)]

        adv_images = np.zeros((num_imgs_per_batch, 3, 224, 224))
        clean_labels = np.zeros(num_imgs_per_batch)
        target_labels = np.zeros(num_imgs_per_batch)
        clean_pred_labels = np.zeros(num_imgs_per_batch)
        adv_pred_labels = np.zeros(num_imgs_per_batch)

        for img_i in range(num_imgs_per_batch):
            image, clean_label = x_val[img_i, :, :, :], y_val[img_i]

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


            adv_images[img_i, :, :, :] = adversarial
            clean_labels[img_i] = clean_label
            target_labels[img_i] = target_label
            clean_pred_labels[img_i] = np.argmax(fmodel.predictions(image))


            print('Iter, Clean, Clean_pred, Adv, Adv_pred:', img_i, clean_label, clean_pred_labels[img_i], target_label,
                  adv_pred_labels[img_i])

        # save adversarial images and target labels
        np.savez(adv_save_dir + 'eps%.2f'%epsilon + '/ResNet50_graybox_l2_advs_%i.npz'%(val_batch + 1),
                 adv_images=adv_images,
                 clean_labels=clean_labels,
                 target_labels=target_labels,
                 clean_pred_labels=clean_pred_labels,
                 adv_pred_labels=adv_pred_labels
                 )

        print('%i of %i completed; adversarial images saved!'%(val_batch, num_batches))
