# Prerequisites

**Please ensure you have prepared the environment ([install.md](install.md)) and the nuScenes dataset ([prepare_datasets.md](prepare_datasets.md)).**

# Train and Test

Train ST-Occ with 8 GPUs:
```shell
bash tools/dist_train.sh occupancy_configs/st_occ/stocc-r50-256x704-36e.py 8
```
Note: the released model is trained with a total batch size of 48 (8 GPUs x 6 samples per GPU).

Evaluate ST-Occ (mIoU) with 8 GPUs:
```shell
bash tools/dist_test.sh occupancy_configs/st_occ/stocc-r50-256x704-36e.py path/to/ckpt.pth 8
```

Evaluate the temporal consistency (mSTCV) metrics in addition to mIoU:
```shell
bash tools/dist_test.sh occupancy_configs/st_occ/stocc-r50-256x704-36e_flicker_eval.py path/to/ckpt.pth 8
```
