# Improving the robustness of ImageNet classifiers with an episodic memory and a shape bias
The code here can be used to reproduce the results reported in the following paper:

## Requirements

## Replication
A successful replication would involve the following steps:

1. Pre-process the raw ImageNet training and validation images by running `imagenet_preprocess_train.py` and `imagenet_preprocess_val.py`, respectively, e.g.:
```
python3 imagenet_preprocess_train.py --source_dir /SOURCE/PATH/ --target_dir /TARGET/PATH/
```

2. Make and store the cache by running `make_cache.py`, e.g.:
```
python3 make_cache.py --train_data_dir /TRAIN/DATA/PATH/ --mem_save_dir /CACHE/SAVE/PATH/ --layer 'activation_46'
```

3. Generate black-box and gray-box adversarial images by running ??, e.g.:
