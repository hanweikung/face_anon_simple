#!/usr/bin/env bash
ENVNAME="fams"

eval "$(conda shell.bash hook)"
conda create --name $ENVNAME --no-default-packages -y python=3.8
conda activate $ENVNAME
pip install torch==2.1 torchvision diffusers["torch"]==0.25.0 transformers accelerate ftfy tensorboard datasets xformers bitsandbytes
