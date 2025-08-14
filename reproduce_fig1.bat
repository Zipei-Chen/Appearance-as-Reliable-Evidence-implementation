@echo off
setlocal enabledelayedexpansion

REM 固定 recording_name 值
set "current_item=MPH1Library_00034_01"

REM 激活 conda 环境
call conda activate ARE

REM 更新 a.yaml 文件

powershell -Command "(gc './configs/dataset/prox.yaml') -replace 'dataset_name: .*', 'dataset_name: prox_!current_item!' | Set-Content './configs/dataset/prox.yaml'"
powershell -Command "(gc './configs/dataset/prox.yaml') -replace 'recording_name: .*', 'recording_name: !current_item!' | Set-Content './configs/dataset/prox.yaml'"

powershell -Command "(gc './configs/dataset/prox_add_scene.yaml') -replace 'dataset_name: .*', 'dataset_name: prox_!current_item!' | Set-Content './configs/dataset/prox_add_scene.yaml'"
powershell -Command "(gc './configs/dataset/prox_add_scene.yaml') -replace 'recording_name: .*', 'recording_name: !current_item!' | Set-Content './configs/dataset/prox_add_scene.yaml'"
echo updated yaml file: dataset_name and recording_name = !current_item!

REM 执行 Python 命令
python 1_train_background_only_prox.py
python 2_train_total_prox.py
python 3_visualize_reulst_prox.py


echo finish
pause