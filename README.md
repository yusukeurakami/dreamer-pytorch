Dreamer implementation in PyTorch
======

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

## Dreamer
This repo implements the Dreamer algorithm from [Dream to Control: Learning Behaviors By latent Imagination](https://arxiv.org/pdf/1912.01603.pdf) based on the [PlaNet-Pytorch](https://github.com/Kaixhin/PlaNet). It has been confirmed working on the DeepMind Control Suite/MuJoCo environment. Hyperparameters have been taken from the paper.

## Installation
To install all dependencies with Anaconda run `conda env create -f conda_env.yml` and use `source activate dreamer` to activate the environment. 

## Training (e.g. DMC walker-walk)
```bash
python main.py --algo dreamer --env walker-walk --action-repeat 2 --id name-of-experiement
```

For best performance with DeepMind Control Suite, try setting environment variable `MUJOCO_GL=egl` (see instructions and details [here](https://github.com/deepmind/dm_control#rendering)).

Use Tensorboard to monitor the training.

`tensorboard --logdir results`

## Results
Results and pretrained models can be found in the [releases](https://github.com/Kaixhin/PlaNet/releases).

The performances are compared with the other SoTA algorithms as follows (Note! Tested once using seed 0.)
* [State-SAC](https://github.com/denisyarats/pytorch_sac)
* [PlaNet-PyTorch](https://github.com/Kaixhin/PlaNet)
* [SAC-AE](https://github.com/denisyarats/pytorch_sac_ae)
* [SLAC](https://github.com/ku2482/slac.pytorch)
* [CURL](https://github.com/MishaLaskin/curl)
* [Dreamer (tensorflow2 implementation)](https://github.com/danijar/dreamer)
<!-- 
![finger-spin](imgs/finger-spin.png) -->

<p align="center">
  <img width="800" src="./imgs/finger-spin.png">
  <img width="800" src="./imgs/walker-walk.png">
  <img width="800" src="./imgs/cheetah-run.png">
  <img width="800" src="./imgs/cartpole-swingup.png">
  <img width="800" src="./imgs/reacher-easy.png">
  <img width="800" src="./imgs/ball_in_cup-catch.png">
</p>


## Links
- [Dream to Control: Learning Behaviors By latent Imagination](https://ai.googleblog.com/2020/03/introducing-dreamer-scalable.html)
- [google-research/dreamer](https://github.com/google-research/dreamer)
