# Improving the robustness of ImageNet classifiers with an episodic memory and a shape bias
The code here can be used to reproduce the results reported in the following paper:

## Requirements
* Keras >= 2.2.4
* scikit-learn >= 0.20.3 
* opencv-python >= 3.4.2.17 
* foolbox >= 1.8.0
* image-classifiers >= 0.2.1
* ImageNet training and validation data in their standard directory structure.

We recommend the installation of the required packages via `pip3` inside a virtual environment.

## Replication
A successful replication would involve the following steps:

1. Pre-process the raw ImageNet training and validation images by running `imagenet_preprocess_train.py` and `imagenet_preprocess_val.py`, e.g.:
```
python3 imagenet_preprocess_train.py --source_dir /SOURCE/PATH/ --target_dir /TARGET/PATH/
```

2. Make and store the cache by running `make_cache.py`, e.g.:
```
python3 make_cache.py --train_data_dir /TRAIN/DATA/PATH/ --mem_save_dir /CACHE/SAVE/PATH/ --layer 'activation_46'
```

3. For experiments in section 4.2, also compress the cache by running `pca_cache.py` and `kmeans_cache.py`, e.g.:
```
python3 pca_cache.py --mem_save_dir /CACHE/SAVE/PATH/ --layer 'activation_46' --reduce_factor 8
```

4. Generate black-box and gray-box adversarial images by running `generate_blackbox_adversarial.py` and `generate_graybox_adversarial.py`, e.g.:
```
python3 generate_graybox_adversarial.py --base_dir /BASE/PATH/ --epsilon 0.06
```

5. Evaluate the clean, gray-box and black-box adversarial accuracies of the cache models by running `evaluate_clean.py`, `evaluate_graybox.py`, and `evaluate_blackbox.py`, respectively, e.g.:
```
python3 evaluate_clean.py --
```

6. Run white-box attacks against the cache models by running `generate_whitebox_adversarial.py`, e.g.:
```
python3 generate_whitebox_adversarial.py --
```

