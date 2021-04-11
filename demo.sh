#!/bin/sh

#---------------------------------------VSpSR--------------------------------------#
# demo

# to evaluate with the x4 model
#nohup python -u evaluate.py input_dir test --output_dir output --first_k 1 --number 10 --model ./checkpoint/checkpoint_x4.pth --scale 4 >out0 &

# to evaluate with the x8 model
#nohup python -u evaluate.py input_dir test --output_dir output --first_k 1 --number 10 --model ./checkpoint/checkpoint_x8.pth --scale 8 >out1 &


# train

# to train a x4 model
#nohup python -u main.py --lr 1e-5 --output_dir logs/train --dataset_root NTIRE2021 --crop_size 48 --scale 4 --variational_w --upsample --GAN --VGG >out2 &

# to train a x8 model
#nohup python -u main.py --output_dir logs/train --dataset_root NTIRE2021 --crop_size 32 --scale 8 --beta 0.1 --upsample_mode bicubic --variational_w --upsample --GAN --VGG >out3 &
