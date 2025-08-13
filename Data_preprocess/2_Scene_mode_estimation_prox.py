import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from os.path import join as pjoin
from PIL import Image
import torchvision.transforms as T

import json

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from joblib import Parallel, delayed
import pandas as pd
import matplotlib.pyplot as plt

test_list = ['MPH1Library_00034_01', 'N0Sofa_00034_01', 'N0Sofa_00034_02', 'N0Sofa_00141_01',
                 'N0Sofa_00145_01', 'N3Library_00157_01', 'N3Library_00157_02', 'N3Library_03301_01',
                 'N3Library_03301_02', 'N3Library_03375_01', 'N3Library_03375_02', 'N3Library_03403_01',
                 'N3Library_03403_02', 'N3Office_00034_01', 'N3Office_00139_01', 'N3Office_00150_01',
                 'N3Office_00153_01', 'N3Office_00159_01', 'N3Office_03301_01']

root_dir = 'C:/RoHM/datasets/PROX/'

list = {}
mask_list = {}
depth_lists = {}
depth_pred_lists = {}
cam2world = {}
scenes = []
color_dir = pjoin(root_dir, "recordings")

# 聚类参数
num_bins = 40  # 聚类簇数，可以根据需求调整

bins = np.linspace(0, 255, num_bins) # 从0到1分成10个区间，包括边界


def process(i):
    print(f"正在处理第 {i + 1} 个片段的聚类...")
    vectors = image_total[:, i, :]

    center = np.zeros(3)

    for j in range(3):
        vectors_j = vectors[:, j]
        hist, bin_edges = np.histogram(vectors_j, bins=bins)
        min_index = np.argmax(hist)
        if min_index > 0:
            center[j] = bin_edges[min_index-1] + (bin_edges[min_index] - bin_edges[min_index-1]) / 2
        else:
            center[j] = bin_edges[min_index] / 2

    return center



for scene in test_list:
    scene_list = []
    scenes.append(scene)
    for sample in os.listdir(pjoin(pjoin(color_dir, scene), "Color")):
        scene_list.append(pjoin(pjoin(pjoin(color_dir, scene), "Color"), sample))
    list[scene] = scene_list
    mask_list_tmp = sorted(glob.glob(pjoin(color_dir, scene, "mask_predict_deeplab") + "/*.png"))
    mask_list[scene] = mask_list_tmp

h = int(1080)
w = int(1920)
for scene in test_list:

    image_list = []

    # 初始化结果存储，记录每次聚类中误差最小的簇的中心
    best_min_cluster_centers = []

    for k, filename in enumerate(list[scene]):
        print(f'Progress {k+1}/{len(list[scene])}: {filename}')

        original_image = np.array(Image.open(filename).convert('RGB').resize([w, h]))
        mask_image = np.array(Image.open(mask_list[scene][k]).resize([w, h]))

        # image_random = original_image * (mask_image != 0)[..., None] + np.random.randint(0, 256, original_image.shape, dtype=np.uint8) * (mask_image == 0)[..., None]
        image_random = original_image + np.random.randint(0, 256, original_image.shape, dtype=np.uint8) * (mask_image == 0)[..., None]
        image_list.append(np.clip(image_random, 0, 255))
        # image_list.append(np.array(Image.open(filename).convert('RGB').resize([w , h])))
    image_total = np.stack(image_list, axis=0)
    image_total = image_total.reshape(image_total.shape[0], -1, 3)

    '''temporal_info = image_total[:, 1038000, :].mean(-1)
    x = np.arange(temporal_info.shape[0])

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(x, temporal_info, linestyle='-', color='b')
    plt.xlabel('Time')
    plt.ylabel('Value')
    # plt.title('Line Plot')
    plt.grid(False)
    plt.show()'''

    result = Parallel(n_jobs=16)(delayed(process)(i) for i in range(image_total.shape[1]))

    # for i in range(image_total.shape[1]):
    #     process(i)

    background = np.stack(result, axis=0).reshape(h, w, 3)

    output_dir = pjoin(pjoin(color_dir, scene, "Background_predicted_histogram"))
    os.makedirs(output_dir, exist_ok=True)
    background_image = Image.fromarray(np.clip(background.astype(np.uint8), 0, 255)).resize((1920, 1080), Image.NEAREST).save(pjoin(output_dir, "background.jpg"))


