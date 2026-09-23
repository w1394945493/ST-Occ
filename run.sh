
# 就地编译mmdet3d
python setup.py build_ext --inplace

# 数据集生成
python tools/create_data_bevdet.py

PYTHONPATH="$(pwd)"  \
python /vepfs-mlp2/c20250502/haoce/wangyushen/ST-Occ/tools/test.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/ST-Occ/occupancy_configs/st_occ/stocc-r50-256x704-36e_flicker_eval_custom.py \
    /c20250502/wangyushen/Weights/st-occ/stocc-r50-256x704.pth \
    --work-dir out/val \