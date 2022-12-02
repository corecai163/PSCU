# PSCU
Parametric Surface Constrained Upsampler Network for Point Cloud

# Pretrained Model and Results


# Train
With 2 GPU:

python3 -m torch.distributed.launch --nproc_per_node=2 multi_train.py

# Test

python3 test.py

# P2F and Uniformity
The p2f evaluation code is from PUGCN.
You may need to compile it by running compile.sh first and then eval_pu1k.sh

To show the p2f, modify and run show_p2f.py
It will also calculate the Uniformity Score.

# Generate colored PCD based on P2F
Run gen_pcd_distance2rgb.py
