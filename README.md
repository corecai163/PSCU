# PSCU
Parametric Surface Constrained Upsampler Network for Point Cloud

# Environment

Pytorch 1.12.0 with Nvidia GPUs

Setup Libs

Install pointnet2_ops_lib and Chamfer3D:

python3 setup.py install


# Data and Results

https://drive.google.com/drive/folders/1Yz9WfAJy145hmD-F1MUvHsjwr6doaOTn?usp=sharing

# Pretrained Model on PU1K

outpath/checkpoints/ckpt-best.pth

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

# Generate color PCD based on P2F
Run gen_pcd_distance2rgb.py

# Citation
If our method and results are useful for your research, please consider citing:

```
@inproceedings{PSCU,
    title={Parametric Surface Constrained Upsampler Network for Point Cloud},
    author={Pingping Cai, Zhenyao Wu, Xinyi Wu, Song Wang},
    booktitle={AAAI},
    year={2023},
}
```

# Acknowledgement
Some codes are borrowed from https://github.com/AllenXiangX/SnowflakeNet and https://github.com/guochengqian/PU-GCN
