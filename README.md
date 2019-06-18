# Improving the robustness of ImageNet classifiers with an episodic memory and a shape bias
The code here can be used to reproduce the results reported in the following paper:

Orhan AE, Lake BM (2019) Improving the robustness of ImageNet classifiers using elements of human visual cognition. arxiv:???

## Requirements
* Keras >= 2.2.4
* scikit-learn >= 0.20.3 
* opencv-python >= 3.4.2.17 
* foolbox >= 1.8.0
* image-classifiers >= 0.2.1
* ImageNet training and validation data in their standard directory structure.
* ImageNet-C data in its standard directory structure (for ImageNet-C results only).

We recommend installing the required packages via `pip3` inside a virtual environment.

## Replication
A successful replication would involve the following steps:

1. Pre-process the raw ImageNet training and validation images by running `imagenet_preprocess_train.py` and `imagenet_preprocess_val.py`, e.g.:
```
python3 imagenet_preprocess_train.py --source_dir /SOURCE/DIR/ --target_dir /TARGET/DIR/
```

2. Make and store the cache by running `make_cache.py`, e.g.:
```
python3 make_cache.py --train_data_dir /TRAIN/DATA/DIR/ --mem_save_dir /CACHE/SAVE/DIR/ --layer 'activation_46'
```

3. For experiments in section 4.2, also compress the cache by running `pca_cache.py` and `kmeans_cache.py`, e.g.:
```
python3 pca_cache.py --mem_save_dir /CACHE/SAVE/DIR/ --layer 'activation_46' --reduce_factor 8
```

4. Generate black-box and gray-box adversarial images by running `generate_blackbox_adversarial.py` and `generate_graybox_adversarial.py`, e.g.:
```
python3 generate_graybox_adversarial.py --base_dir /BASE/DIR/ --epsilon 0.06
```

5. Evaluate the clean, gray-box and black-box adversarial accuracies of the cache models by running `evaluate_clean.py`, `evaluate_graybox.py`, or `evaluate_blackbox.py` respectively, e.g.:
```
python3 evaluate_clean.py --val_data_dir /VAL/DATA/DIR/ --mem_save_dir /CACHE/SAVE/DIR/ --layer 'activation_46' --theta 50.0
```

6. Run white-box attacks against the cache models by running `generate_whitebox_adversarial.py`, e.g.:
```
python3 generate_whitebox_adversarial.py --base_dir /BASE/DIR/ --mem_save_dir /CACHE/SAVE/DIR/ --layer 'activation_46' --theta 50.0 --epsilon 0.06
```

