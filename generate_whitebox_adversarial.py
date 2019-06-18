"""Run white-box adversarial attacks against cache models.
"""
import argparse
import numpy as np
from keras.backend import set_learning_phase
from scipy.io import loadmat
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from foolbox.models import KerasModel
from foolbox.attacks import RandomStartProjectedGradientDescentAttack
from foolbox.criteria import TargetClass
from NewKerasModel import MultiInputKerasModel
from utils import Ltwo

parser = argparse.ArgumentParser(description='Evaluate cache models on gray-box adversarial images')
parser.add_argument('--base_dir', type=str, help='Base directory')
parser.add_argument('--mem_save_dir', type=str, help='directory where the caches are stored')
parser.add_argument('--layer', type=str, default='activation_46',
                    choices=['fc1000', 'avg_pool', 'activation_46', 'activation_43'],
                    help='which layer to use for embedding')
parser.add_argument('--theta', type=float, default=50.0, help='theta hyper-parameter (inverse temperature)')
parser.add_argument('--epsilon', type=float, default=0.06, help='Perturbation size (normalized l2-norm)')

args = parser.parse_args()

val_data_dir = args.base_dir + 'processed_val/'
mem_save_dir = args.mem_save_dir
epsilon = args.epsilon
theta = args.theta

n_classes = 1000
set_learning_phase(0)

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

# Load cache
data = np.load(mem_save_dir + 'ResNet50_cache_all_layer%i.npz'%mem_layer)

if mem_layer==176:
    mem_keys, mem_vals = data['mem_keys'], data['mem_vals']
else:
    mem_keys1, mem_vals1 = data['mem_keys'][::2, :], data['mem_vals'][::2, :]   # evens
    mem_keys2, mem_vals2 = data['mem_keys'][1::2, :], data['mem_vals'][1::2, :]  # odds
    mem_keys = np.concatenate((mem_keys1, mem_keys2[::2, :]), axis=0)
    mem_vals = np.concatenate((mem_vals1, mem_vals2[::2, :]), axis=0)

# normalize keys
mem_keys = mem_keys / np.linalg.norm(mem_keys, axis=1, keepdims=True)

print('Key and value matrix shapes: ', mem_keys.shape, mem_vals.shape)
print('Successfully loaded the cache.')

# Define model
model = ResNet50(weights='imagenet')

if mem_layer == 164 or mem_layer == 154:
    output = GlobalAveragePooling2D()(model.layers[mem_layer].output)
else:
    output = model.layers[mem_layer].output

print('Cache layer:', model.layers[mem_layer].name)

# Define memory
mem = Model(inputs=model.input, outputs=output)

test_mem = mem(model.input)
test_mem_norm = Lambda(lambda x: K.sqrt(K.sum(x**2, axis=1, keepdims=True)))(test_mem)
test_mem_norm = Lambda(lambda x: K.tile(x, (1, mem_keys.shape[-1])))(test_mem_norm)
test_mem = Lambda(lambda x: x[0] / x[1])([test_mem, test_mem_norm])

key_place = Input(batch_shape=(mem_keys.shape[1], mem_keys.shape[0]))
val_place = Input(batch_shape=(mem_vals.shape[0], mem_vals.shape[1]))

similarities = Lambda(lambda x: K.exp(theta * K.dot(x[0], x[1]) ))([test_mem, key_place])
p_mem = Lambda(lambda x: K.log(K.dot(x[0], x[1])))([similarities, val_place])  # these need to be logits.

cache_model = Model(inputs=[model.input, key_place, val_place], outputs=p_mem)
cache_model.summary()

preprocessing = (np.array([103.939, 116.779, 123.68]), 1)
fmodel_cache = MultiInputKerasModel(cache_model, mem_keys.T, mem_vals, bounds=(0, 255), preprocessing=preprocessing)

num_batches = 10
num_imgs_per_batch = 5000
val_accs_mem = np.zeros(num_batches)
n_imgs_small = 150  # run only this many attacks because of computational cost

val_ground_truth = np.loadtxt('ILSVRC2012_validation_ground_truth.txt', usecols=1)

for val_batch in range(1):
    x_val = loadmat(val_data_dir + 'val_batch_%i'%(val_batch + 1))['all_imgs']
    y_val = val_ground_truth[(val_batch * num_imgs_per_batch):((val_batch + 1) * num_imgs_per_batch)]

    clean_labels = np.zeros(n_imgs_small)
    target_labels = np.zeros(n_imgs_small)
    clean_pred_labels = np.zeros(n_imgs_small)
    adv_pred_labels = np.zeros(n_imgs_small)

    for img_i in range(n_imgs_small):
        image, clean_label = x_val[img_i, :, :, :], y_val[img_i]

        target_label = np.random.choice(np.setdiff1d(np.arange(n_classes), clean_label))
        attack = RandomStartProjectedGradientDescentAttack(model=fmodel_cache, criterion=TargetClass(target_label),
                                                           distance=Ltwo)
        adversarial = attack(image[:, :, ::-1], clean_label,
                             binary_search=False, epsilon=epsilon, stepsize=2./255, iterations=10, random_start=True)

        if np.any(adversarial==None):
            adversarial=image[..., ::-1]
            target_label=clean_label
            adv_pred_labels[img_i] = np.argmax(model.predict(preprocess_input(image[np.newaxis, :, :, :])))
            # print('No adversarial!')
        else:
            adv_pred_labels[img_i] = np.argmax(fmodel_cache.predictions(adversarial))


        clean_labels[img_i] = clean_label
        target_labels[img_i] = target_label
        clean_pred_labels[img_i] = np.argmax(model.predict(preprocess_input(image[np.newaxis, :, :, :])))


        print('Iter, Clean, Clean_pred, Adv, Adv_pred: ', img_i, clean_label, clean_pred_labels[img_i], target_label,
              adv_pred_labels[img_i])

mem_acc = np.mean(clean_labels==adv_pred_labels)
print('Mean mem. accuracy:', mem_acc)

np.savez('cache_whitebox_accuracy_' + 'eps%.2f'%epsilon + '_layer%i_theta%2.f.npz'%(mem_layer, theta),
         mem_acc=mem_acc,
         layer=mem_layer,
         theta=theta
         )