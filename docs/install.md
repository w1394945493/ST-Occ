# Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n stocc python=3.8 -y
conda activate stocc
```
Alternatively, create the environment with all auxiliary dependencies pre-installed from
[environment.yml](environment.yml) (you still need steps b, c and e below for torch, mmcv-full
and this repo):
```shell
conda env create -f docs/environment.yml
conda activate stocc
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.12
```

**c. Install mmcv-full (with CUDA ops).**

`mmcv-full` must be compiled with CUDA ops, otherwise inference/training will fail with
`ms_deform_attn_impl_forward: implementation for device cuda not found`.
Build it from source with `FORCE_CUDA=1` (make sure `nvcc` matching your PyTorch CUDA version is available):
```shell
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv-full==1.5.2
```
Verify the CUDA ops are available:
```shell
python -c "from mmcv.ops import get_compiling_cuda_version; print(get_compiling_cuda_version())"
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.24.0
pip install mmsegmentation==0.24.0
```

**e. Install ST-Occ from source code.**
```shell
git clone https://github.com/matthew-leng/ST-Occ.git
cd ST-Occ
pip install -e .
```

**f. Prepare pretrained models.**
```shell
cd ST-Occ
mkdir ckpts && cd ckpts
# ImageNet pretrained ResNet-50
wget https://download.pytorch.org/models/resnet50-0676ba61.pth
# Depth pretrained model (from FB-OCC)
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r50_256x705_depth_pretrain.pth
```
