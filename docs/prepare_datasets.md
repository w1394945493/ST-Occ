# Prepare Datasets

## NuScenes

Download nuScenes V1.0 full dataset data from the [official website](https://www.nuscenes.org/download).

For the occupancy prediction task, you also need to download the Occ3D-nuScenes occupancy annotations (`gts`) from
https://github.com/Tsinghua-MARS-Lab/Occ3D

**Prepare nuScenes info files**

*We generate custom annotation files which are different from the original BEVDet's: besides the per-frame
annotations, scene-level global information (ego trajectory, scene bounds, per-frame ego-to-global transforms)
is added for the scene-level spatiotemporal memory.*

```shell
python tools/create_data_bevdet.py
```

This generates `bevdetv4-nuscenes_infos_{train,val}.pkl` and then
`bevdetv4-nuscenes_infos_global_{train,val}.pkl` (the ones used by the ST-Occ configs).

**Folder structure**
```
ST-Occ
├── mmdet3d/
├── tools/
├── occupancy_configs/
├── ckpts/
│   ├── resnet50-0676ba61.pth
│   ├── r50_256x705_depth_pretrain.pth
├── data/
│   ├── nuscenes/
│   │   ├── gts/  # ln -s Occ3D occupancy gts to this location
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   │   ├── bevdetv4-nuscenes_infos_global_train.pkl
│   │   ├── bevdetv4-nuscenes_infos_global_val.pkl
```
