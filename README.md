# DualPure: An Efficient Adversarial Purification Method for Speech Command Recognition

This is the PyTorch implementation of the paper "DualPure: An Efficient Adversarial Purification Method for Speech Command Recognition" published in Interspeech'2024.

## Code preparation
Since our codes are based on [AudioPure](https://github.com/cychomatica/AudioPure), you need to  ```git clone https://github.com/cychomatica/AudioPure``` first.

Then replace part of the codes with our release.

## Environment
Create a new conda environment of Python 3.9 called DualPure.
```
conda create -n DualPure python=3.9
conda activate DualPure
pip install -r requirements.txt
```

## Datasets 
- [SC09_100](https://drive.google.com/drive/folders/1_Y89ieKZ4DRtDS9NABmg-7StD4d7umh_)
- [SC09 test](https://drive.google.com/drive/folders/1NT4J4c8RaVEnRto_I9bD9jVokKSS7ACz)
## Pretrained Models
SCR models: 
- [ResNet28_8_64](https://drive.google.com/file/d/1b-YaiprEKx-dMXGnJsi7IHblzUOJlDGC/view?usp=drive_link)
- [VGG19](https://drive.google.com/file/d/1GBeralpVBtdetWfzGWwsfr9yKA57TZfY/view?usp=drive_link)
- [WiderResNet28_10](https://drive.google.com/file/d/1CLIoZ0j1OgJ0J3d939v-gRhnUHewQIrN/view?usp=drive_link)

Purifier: 
- [Diffwave](https://drive.google.com/file/d/1emYZOmDg6NYSdsnG0w38mSKcYaPFYQm6/view?usp=drive_link)
- [Diffspec](https://drive.google.com/file/d/14aoldbx7k9M9KSa6Po8VC3dGfVR-wHAs/view?usp=drive_link) 

## Perform defenses
```
python adaptive_attack_eval.py \
        --data_path {dataset_path} \
        --classifier_path {speech_command_model_path} \
        --diffwav_path {diffwav_model_path} \
        --diffspec_path {diffspec_model_path} \
        --diffusion_type ddpm \
        --attack PGD \
        --defense DualPure \
        --t 2 \
        --sample_step 1 \
        --eot_attack_size 1 \
        --eot_defense_size 1 \
        --bound_norm linf \
        --eps 0.002 \
        --max_iter_1 30 \
        --gpu 0 
```

## Citation

## Acknowledgement
- We thank the excellent work [AudioPure](https://github.com/cychomatica/AudioPure).
- https://github.com/lmnt-com/diffwave.
