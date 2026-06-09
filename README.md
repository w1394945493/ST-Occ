# ST-Occ: Occupancy Learning with Spatiotemporal Memory

[![arXiv](https://img.shields.io/badge/arXiv-2508.04705-b31b1b.svg)](https://www.arxiv.org/abs/2508.04705)&nbsp;

<p align="center">
  <img src="assets/demo_scene.jpg" width="720">
</p>

**ST-Occ** is a scene-level occupancy representation learning framework that effectively learns the spatiotemporal feature with temporal consistency.


<p align="center">
  <img src="assets/method.png" width="720">
</p>

## Getting Started
- [Installation](docs/install.md)
- [Prepare Dataset](docs/prepare_datasets.md)
- [Training, Eval, Visualization](docs/start.md)

## Model Zoo

| Backbone | Method | Lr Schd | mIoU | mSTCV |  Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| R50 | ST-Occ | 28ep | 42.13 | 8.68 |[config](occupancy_configs/st_occ/stocc-r50-256x704-36e.py) |[model](TODO)|

To additionally evaluate the temporal-consistency metric mSTCV (mean SpatioTemporal Classification
Variability), use
[stocc-r50-256x704-36e_flicker_eval.py](occupancy_configs/st_occ/stocc-r50-256x704-36e_flicker_eval.py)
with the same checkpoint (see [docs/start.md](docs/start.md)).


## Acknowledgement

Many thanks to these excellent open source projects:

- [FB-OCC](https://github.com/NVlabs/FB-BEV), [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [BEVDet](https://github.com/HuangJunJie2017/BEVDet), [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D), [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy), [SoloFusion](https://github.com/Divadi/SOLOFusion)

## Citation

Consider citing our paper if you find our paper is useful for your research:

```
@inproceedings{leng2025occupancy,
  title={Occupancy learning with spatiotemporal memory},
  author={Leng, Ziyang and Yang, Jiawei and Yi, Wenlong and Zhou, Bolei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={26569--26578},
  year={2025}
}
```