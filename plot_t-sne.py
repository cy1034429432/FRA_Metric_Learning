# -*- coding: utf-8 -*-
# Author: Yu Chen
# Data: 2026/3/12 16:49
# Email: yu2000.chen@connect.polyu.hk

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from model_utils import SiameseNetwork

import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'



excel_file_name = "Transformer_FRA_indicator_Norm.xlsx"
data_sheet_name = "FRA_indicator"
model = SiameseNetwork(model_name="CNN")
checkpoint = torch.load("logs/CNN_none/best_model.pth", map_location=torch.device('cuda:0'))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
tsne = TSNE(n_components=2, init='pca', random_state=0)
with torch.no_grad():
    df_data = pd.read_excel(excel_file_name, sheet_name=data_sheet_name)

    cols = [
        "ssre_l", "ssre_m", "ssre_h",
        "sd_l", "sd_m", "sd_h",
        "rou_l", "rou_m", "rou_h",
        "mm_l", "mm_m", "mm_h",
        "ed_l", "ed_m", "ed_h",
        "e_l", "e_m", "e_h",
        "delta_l", "delta_m", "delta_h",
        "dabs_l", "dabs_m", "dabs_h",
        "cc_l", "cc_m", "cc_h",
        "asle_l", "asle_m", "asle_h",
    ]

    all_sample_data = torch.tensor(np.array(df_data[cols]), dtype=torch.float32).to(device).reshape(-1, 1, 30)
    _, all_sample_feature, _ = model(all_sample_data, all_sample_data)
    all_sample_feature = all_sample_feature.cpu().numpy().reshape(-1, 10)
    all_sample_feature_tsne = tsne.fit_transform(all_sample_feature)

    #fig_tsne = plt.figure(figsize=(10, 8), dpi=120)
    fig_tsne = plt.figure(figsize=(3.5, 3), dpi=300)
    fault_types = ["AD", "DSV", "IDSC", "RD", "Normal"]
    colors = ['#1E90FF',  # 蓝色 (Dodger Blue)
              '#FFA500',  # 橙色 (Orange)
              '#FF0000',  # 红色 (Red)
              '#32CD32',  # 绿色 (Lime Green)
              '#800080']  # 紫色 (Purple)

    markers = ['o', 's', '^', 'D', '*']  # 不同形状辅助区分
    alpha = 0.7  # 轻微透明，避免重叠太严重
    s = 40  # 点的大小

    # 一次性循环绘制所有类别（更简洁、可维护）
    for code in range(5):
        mask = (df_data["fault_type_code"] == code)
        plt.scatter(
            all_sample_feature_tsne[mask, 0],
            all_sample_feature_tsne[mask, 1],
            c=colors[code],
            marker=markers[code],
            s=s,
            alpha=alpha,
            edgecolor='white',  # 白色边框让点更立体
            linewidth=0.5,
            label=f"{fault_types[code]}"
        )

    # 美化设置
    #plt.title("t-SNE Visualization of Transformer Fault Features", fontsize=16, pad=15)
    plt.xlabel("t-SNE dimension 1", fontsize=9)
    plt.ylabel("t-SNE dimension 2", fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(
        title="Fault Type",
        loc='best',  # 或 'upper right', 'lower left' 等
        fontsize=8,
        title_fontsize=8,
        frameon=True,
        shadow=False,
        fancybox=False
    )
    # 紧凑布局
    plt.tight_layout()
    plt.savefig("tsne_fault_features.pdf", format="pdf", bbox_inches="tight")
    plt.close()