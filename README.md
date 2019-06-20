# Improving the robustness of ImageNet classifiers with an episodic memory and a shape bias
The code here can be used to reproduce the results reported in the following paper:

Orhan AE, Lake BM (2019) [Improving the robustness of ImageNet classifiers using elements of human visual cognition](https://sites.google.com/view/eminorhan). arXiv:1905.13715.

## Requirements
* Keras >= 2.2.4
* scikit-learn >= 0.20.3 
* opencv-python >= 3.4.2.17 
* foolbox >= 1.8.0
* image-classifiers >= 0.2.1
* ImageNet training and validation data in their standard directory structure.
* [ImageNet-C](https://github.com/hendrycks/robustness) data in its standard directory structure (for ImageNet-C results only).

We recommend installing the required packages via `pip3` inside a virtual environment.

## Replication
A successful replication would involve the following steps:

1. Pre-process the raw ImageNet training and validation images by running [`imagenet_preprocess_train.py`](https://github.com/eminorhan/robust-vision/blob/master/imagenet_preprocess_train.py) and [`imagenet_preprocess_val.py`](https://github.com/eminorhan/robust-vision/blob/master/imagenet_preprocess_val.py), e.g.:
```
python3 imagenet_preprocess_train.py --source_dir /SOURCE/DIR/ --target_dir /TARGET/DIR/
```

2. Make and store the cache by running [`make_cache.py`](https://github.com/eminorhan/robust-vision/blob/master/make_cache.py), e.g.:
```
python3 make_cache.py --train_data_dir /TRAIN/DATA/DIR/ --mem_save_dir /CACHE/SAVE/DIR/ --layer 'activation_46'
```

3. For experiments in section 4.2, also compress the cache by running [`pca_cache.py`](https://github.com/eminorhan/robust-vision/blob/master/pca_cache.py) and [`kmeans_cache.py`](https://github.com/eminorhan/robust-vision/blob/master/kmeans_cache.py), e.g.:
```
python3 pca_cache.py --mem_save_dir /CACHE/SAVE/DIR/ --layer 'activation_46' --reduce_factor 8
```

4. Generate black-box and gray-box adversarial images by running [`generate_blackbox_adversarial.py`](https://github.com/eminorhan/robust-vision/blob/master/generate_blackbox_adversarial.py) and [`generate_graybox_adversarial.py`](https://github.com/eminorhan/robust-vision/blob/master/generate_graybox_adversarial.py), e.g.:
```
python3 generate_graybox_adversarial.py --base_dir /BASE/DIR/ --epsilon 0.06
```

5. Evaluate the clean, gray-box and black-box adversarial accuracies of the cache models by running [`evaluate_clean.py`](https://github.com/eminorhan/robust-vision/blob/master/evaluate_clean.py), [`evaluate_graybox.py`](https://github.com/eminorhan/robust-vision/blob/master/evaluate_graybox.py), or [`evaluate_blackbox.py`](https://github.com/eminorhan/robust-vision/blob/master/evaluate_blackbox.py) respectively, e.g.:
```
python3 evaluate_clean.py --val_data_dir /VAL/DATA/DIR/ --mem_save_dir /CACHE/SAVE/DIR/ --layer 'activation_46' --theta 50.0
```

6. Run white-box attacks against the cache models by running [`generate_whitebox_adversarial.py`](https://github.com/eminorhan/robust-vision/blob/master/generate_whitebox_adversarial.py), e.g.:
```
python3 generate_whitebox_adversarial.py --base_dir /BASE/DIR/ --mem_save_dir /CACHE/SAVE/DIR/ --layer 'activation_46' --theta 50.0 --epsilon 0.06
```

7. For experiments in section 4.2, evaluate the clean and gray-box adversarial accuracies of the compressed cache models by running [`evaluate_clean_compressed.py`](https://github.com/eminorhan/robust-vision/blob/master/evaluate_clean_compressed.py) or [`evaluate_graybox_compressed.py`](https://github.com/eminorhan/robust-vision/blob/master/evaluate_graybox_compressed.py) respectively, e.g.:
```
python3 evaluate_clean_compressed.py --val_data_dir /VAL/DATA/DIR/ --mem_save_dir /CACHE/SAVE/DIR/ --layer 'activation_46' --theta 50.0 --reduce_method 'kmeans' --reduce_factor 8
```

8. For the ImageNet-C experiments, first pre-process the raw ImageNet-C data with [`imagenetc_preprocess.py`](https://github.com/eminorhan/robust-vision/blob/master/imagenetc_preprocess.py), then evaluate the cache models by running [`evaluate_imagenetc.py`](https://github.com/eminorhan/robust-vision/blob/master/evaluate_imagenetc.py), e.g.:
```
python3 evaluate_imagenetc.py --val_data_dir /IMAGENETC/DIR/ --mem_save_dir /CACHE/SAVE/DIR/ --layer 'activation_46' --theta 50.0 --corruption 'brightness' --severity '1'
```

9. For the Shape-ResNet-50 results, follow similar steps. The corresponding files for these experiments are in the [`shape-resnet-50`](https://github.com/eminorhan/robust-vision/tree/master/shape-resnet-50) directory. For these experiments, you will need to have torch (>=1.0) and torchvision (>=0.2.2) installed, in addition to the [pre-trained Shape-ResNet-50 models](https://github.com/rgeirhos/texture-vs-shape) kindly provided by Robert Geirhos (which you can get simply by running [`save_pretrained_models.py`](https://github.com/eminorhan/robust-vision/tree/master/shape-resnet-50/save_pretrained_models.py)).
