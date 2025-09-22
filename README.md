# MultiDiff: Multi-Candidiate Reverse Diffusion for Adversarial Defence in Speech Command Recognition

This repostory implements **MultiDiff**, an adversarial defence for speech command recongition that leverages short reverse diffusion reconstruction and multi-candidate selection. For each adversarial input, we employ short reverse diffusion processes from an intermediate timestep to generate multiple candidate reconstructions, then apply a lightweight scoring criterion to identify the best candidate for classification.

Results and Audio Samples can be found [here](https://honourswebsite.vercel.app/).

This work was developed as part of the **Honours project** at the **University of Western Australia**.

## Environment
Create a new conda environment 
```
conda create -n HonoursEnv python=3.9
conda activate HonoursEnv
pip install -r requirements.txt 
```

## Datasets 
To download the dataset and split it into training, validation, and test sets, run `download_speech_commands_dataset.sh`  

## Pretrained Models
### Classifer: 
- [ResNet28_8_64](https://drive.google.com/file/d/1b-YaiprEKx-dMXGnJsi7IHblzUOJlDGC/view?usp=drive_link)
### Diffusion Models:
- [DiffWave](https://github.com/philsyn/DiffWave-unconditional/blob/master/exp/ch256_T200_betaT0.02/logs/checkpoint/1000000.pkl)
- [DiffSpec](https://drive.google.com/file/d/14aoldbx7k9M9KSa6Po8VC3dGfVR-wHAs/view?usp=drive_link)

## Perform defenses

```
python adaptive_attack_eval.py \
   --data_path <path to downloaded dataset> \
   --classifier_path <path to classifier> \
   --diffwav_path <path DiffWave model> \
   --diffspec_path <path to diffSpec model> \   
   --attack PGD \ # {PGD | FAKEBOB}  
   --defense MultiDiff \
   --defense_type wave \
   --diffusion_type ddpm \ 
   --t 50 \
   --sample_step 1 \ 
   --rand_inits 3 \
   --select logit_margin \
   --bound_norm l2 \ # {Linf | l2}
   --eps 0.253 \ # {0.002 | 0.253} 
   --max_iter_1 10 \ # {10, 30, 50, 70, 100}
   --bpda_mode surrogate \ 
   --bpda_t 5 \
   --gpu 0
```




## Acknowledgement
- We thank the excellent works of [AudioPure](https://github.com/cychomatica/AudioPure) and [DualPure](https://github.com/Sec4ai/DualPure).

