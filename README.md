# Ring-defender

## Prerequisites
1. PyTorch + CUDA enabled GPU
2. Scikit-learn
3. (For data collection) Intel CPU (after Skylake) + Linux

## Instructions
1. Use DefenderGAN.py to train DefenderGAN. The best generator will be stored to gans/ directory. See Utils.py for arguments.
``` python DefenderGAN.py --victim rsa --dim 160 --epochs 100 --amp 3.0 --fresh```
2. Use Compressor.py to compress the trained defender
``` python Compressor.py --victim rsa --dim 160 --studentdim 16 --epochs 100```
3. We already provided pre-collected dataset in .pkl files. To collect your own data, follow instructions in lotr/04-crypto-sc/README.md and run lotr-parser.ipynb to parse the collected data.