# Ring-defender

## Prerequisites
1. PyTorch + CUDA enabled GPU
2. Scikit-learn
3. (For data collection) Intel Desktop CPU (after Skylake)

## Instructions
1. Use DefenderGAN.py to train DefenderGAN. The best generator will be stored to gans/ directory. See Utils.py for arguments.
``` python DefensiveML.py --victim rsa --fresh```
3. Use Evaluate.py to evaluate the trained defender with adaptive attackers.
``` python Evaluate.py --victim rsa```
4. We already provided pre-collected dataset in .pkl files. To collect your own data, follow instructions in lotr/04-crypto-sc/README.md and run lotr-parser.ipynb to parse the collected data.