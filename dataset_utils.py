# -*- coding: utf-8 -*-
# Author: Yu Chen
# Data: 2026/3/1 21:10
# Email: yu2000.chen@connect.polyu.hk


from typing import List, Tuple, Optional, Dict
import random
import pandas as pd
import numpy as np
import torch
from sympy.codegen import Print
from torch.utils.data import Dataset, DataLoader


class SFRA_indicator_dataset_train(Dataset):
    """
    基于 FRA 指标的配对数据集（用于相似性 + triplet 学习）

    功能：
    - 从 Excel 读取特征数据与人工构建的正/负样本对
    - 支持对相似/不相似样本对进行 over/undersampling 平衡
    - 为每个样本对动态随机采样 triplet 的正样本与负样本（基于 fault_type_code）

    返回格式：字典，包含两个 anchor + 各自的正负样本（共 6 个特征向量）
    """
    def __init__(self,
                 excel_file_name = "Transformer_FRA_indicator_Norm.xlsx",
                 data_sheet_name = "FRA_indicator",
                 positive_sample_sheet_name = "FRA_positive_sample_pairs",
                 negative_sample_sheet_name = "FRA_negative_sample_pairs",
                 resample = "oversample",    # 'none' | 'oversample' | 'undersample'
                 random_state = 42):
        super(SFRA_indicator_dataset_train, self).__init__()

        # 固定随机种子，保证可复现
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        self.excel_file_name = excel_file_name
        self.data_sheet_name = data_sheet_name
        self.positive_sample_sheet_name = positive_sample_sheet_name
        self.negative_sample_sheet_name = negative_sample_sheet_name

        self.resample_strategy = resample.lower()
        if self.resample_strategy not in ("none", "oversample", "undersample"):
            raise ValueError("resample 必须是 'none' / 'oversample' / 'undersample'")

        # 固定 30 维特征列名
        self.cols = [
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

        # 加载数据 & 重采样
        (self.first_sample_data, self.first_sample_fault_label, self.first_class_rank,
         self.second_sample_data, self.second_sample_fault_label, self.second_class_rank,
         self.label, self.df_data) = self.load_indicator_data()

        # 提前把全量特征表转为 tensor，供 triplet 快速索引（极大提升效率）
        self.all_data_tensor = torch.tensor(
            self.df_data[self.cols].to_numpy(),
            dtype=torch.float32
        )  # shape: (n_samples_total, 30)

        # 方便后续使用
        self.fault_type_series = self.df_data["fault_type_code"]



    def load_indicator_data(self):
        """读取原始配对数据，并根据策略进行相似/不相似样本的重采样"""

        Transformer_Fault_dict = {"AD": 0, "DSV": 1, "ITSC": 2, "RD": 3, "Normal": 4}

        cols = self.cols

        # 读取数据表
        df_data = pd.read_excel(self.excel_file_name, sheet_name=self.data_sheet_name)

        # 读取正负样本对 & 合并正负样本对
        df_pos = pd.read_excel(self.excel_file_name, sheet_name=self.positive_sample_sheet_name)
        df_neg = pd.read_excel(self.excel_file_name, sheet_name=self.negative_sample_sheet_name)
        df_pairs = pd.concat([df_pos, df_neg], ignore_index=True)

        # 故障类别转数字编码
        first_class_fault_label = np.array([Transformer_Fault_dict[x] for x in df_pairs['First_class']])
        second_class_fault_label = np.array([Transformer_Fault_dict[x] for x in df_pairs['Second_class']])

        # 样本在 df_data 中的行号（索引）
        first_class_rank = df_pairs['First_class_rank'].to_numpy()
        second_class_rank = df_pairs['Second_class_rank'].to_numpy()

        # 相似性标签（1=相似，0=不相似）
        label = df_pairs['Whether_similarity'].to_numpy()

        # 根据索引提取特征
        first_sample_data_np = df_data.iloc[first_class_rank][self.cols].to_numpy()
        second_sample_data_np = df_data.iloc[second_class_rank][self.cols].to_numpy()

        # 转换为张量并 reshape 成 (batch, 1, 30)
        first_sample_data = torch.tensor(first_sample_data_np, dtype=torch.float32).unsqueeze(1)
        second_sample_data = torch.tensor(second_sample_data_np, dtype=torch.float32).unsqueeze(1)
        label = torch.tensor(label, dtype=torch.long)

        first_sample_fault_label = torch.tensor(first_class_fault_label, dtype=torch.long)
        second_sample_fault_label = torch.tensor(second_class_fault_label, dtype=torch.long)

        # 下述程序用以应付标签不均衡
        # -------- 重采样策略 --------
        if self.resample_strategy == "none":
            pass

        elif self.resample_strategy == "oversample":
            pos_idx = (label == 1).nonzero(as_tuple=True)[0]
            neg_idx = (label == 0).nonzero(as_tuple=True)[0]
            max_len = max(len(pos_idx), len(neg_idx))

            pos_repeat = pos_idx.repeat(int(np.ceil(max_len / len(pos_idx))))[:max_len]
            neg_repeat = neg_idx.repeat(int(np.ceil(max_len / len(neg_idx))))[:max_len]

            idx_all = torch.cat([pos_repeat, neg_repeat])

            first_sample_data = first_sample_data[idx_all]
            second_sample_data = second_sample_data[idx_all]
            label = label[idx_all]
            first_sample_fault_label = first_class_fault_label[idx_all]
            second_sample_fault_label = second_class_fault_label[idx_all]
            first_class_rank = first_class_rank[idx_all.cpu().numpy()]
            second_class_rank = second_class_rank[idx_all.cpu().numpy()]


        elif self.resample_strategy == "undersample":
            pos_idx = (label == 1).nonzero(as_tuple=True)[0]
            neg_idx = (label == 0).nonzero(as_tuple=True)[0]
            min_len = min(len(pos_idx), len(neg_idx))

            pos_sample = pos_idx[torch.randperm(len(pos_idx))[:min_len]]
            neg_sample = neg_idx[torch.randperm(len(neg_idx))[:min_len]]

            idx_all = torch.cat([pos_sample, neg_sample])

            first_sample_data = first_sample_data[idx_all]
            second_sample_data = second_sample_data[idx_all]
            label = label[idx_all]
            first_sample_fault_label = first_class_fault_label[idx_all]
            second_sample_fault_label = second_class_fault_label[idx_all]
            first_class_rank = first_class_rank[idx_all.cpu().numpy()]
            second_class_rank = second_class_rank[idx_all.cpu().numpy()]

        return (first_sample_data, first_sample_fault_label, first_class_rank,
                second_sample_data, second_sample_fault_label, second_class_rank,
                label, df_data)


    def __len__(self):
        return self.first_sample_data.shape[0]

    def _sample_triplet_indices(self, anchor_rank: int, target_label: int):
        """从全数据集随机抽取一个正样本和一个负样本的索引（排除自身）"""
        # 注意：这里使用的是原始 df_data 的行号（0~N-1）
        mask_same = (self.fault_type_series == target_label) & (self.fault_type_series.index != anchor_rank)
        mask_diff = (self.fault_type_series != target_label) & (self.fault_type_series.index != anchor_rank)

        pos_candidates = self.fault_type_series.index[mask_same].tolist()
        neg_candidates = self.fault_type_series.index[mask_diff].tolist()

        if not pos_candidates:
            raise RuntimeError(f"类别 {target_label} 样本太少，无法为 {anchor_rank} 找到正样本")
        if not neg_candidates:
            raise RuntimeError(f"无法为 {anchor_rank} 找到负样本（几乎所有样本都是同一类？）")

        return random.choice(pos_candidates), random.choice(neg_candidates)


    def __getitem__(self, idx):
        """从全数据集随机抽取一个正样本和一个负样本的索引（排除自身）"""

        # 配对的两个 anchor 特征
        first_feat = self.first_sample_data[idx]  # shape (1, 30)
        second_feat = self.second_sample_data[idx]

        first_fault_label = self.first_sample_fault_label[idx]  # 已为 tensor
        second_fault_label = self.second_sample_fault_label[idx]

        similarity = self.label[idx]

        # 原始数据表中的行号（用于查找同类/不同类样本）
        first_sample_rank = int(self.first_class_rank[idx])  # 确保是 python int
        second_sample_rank = int(self.second_class_rank[idx])

        # ──────────────── Triplet 第一组 ────────────────
        pos_idx_first, neg_idx_first = self._sample_triplet_indices(
            first_sample_rank, first_fault_label.item()
        )
        pos_feat_first = self.all_data_tensor[pos_idx_first].unsqueeze(0)  # (1, 30)
        neg_feat_first = self.all_data_tensor[neg_idx_first].unsqueeze(0)

        pos_fault_label_first = torch.tensor(
            self.fault_type_series.iloc[pos_idx_first], dtype=torch.long
        )
        neg_fault_label_first = torch.tensor(
            self.fault_type_series.iloc[neg_idx_first], dtype=torch.long
        )

        # ──────────────── Triplet 第二组 ────────────────
        pos_idx_second, neg_idx_second = self._sample_triplet_indices(
            second_sample_rank, second_fault_label.item()
        )
        pos_feat_second = self.all_data_tensor[pos_idx_second].unsqueeze(0)
        neg_feat_second = self.all_data_tensor[neg_idx_second].unsqueeze(0)

        pos_fault_label_second = torch.tensor(
            self.fault_type_series.iloc[pos_idx_second], dtype=torch.long
        )
        neg_fault_label_second = torch.tensor(
            self.fault_type_series.iloc[neg_idx_second], dtype=torch.long
        )


        return { "first_feat": first_feat,
                 "first_fault_label": first_fault_label,
                 "second_feat": second_feat,
                 "second_fault_label": second_fault_label,
                 "similarity": similarity,
                 "pos_feat_first": pos_feat_first,
                 "pos_fault_label_first": pos_fault_label_first,
                 "neg_feat_first": neg_feat_first,
                 "neg_fault_label_first": neg_fault_label_first,
                 "pos_feat_second": pos_feat_second,
                 "pos_fault_label_second": pos_fault_label_second,
                 "neg_feat_second": neg_feat_second,
                 "neg_fault_label_second": neg_fault_label_second }



class SFRA_indicator_dataset_test(Dataset):
    """
    自定义 PyTorch Dataset，用于加载 FRA 指标测试数据。
    从 Excel 文件中读取样本对及其标签，并转换为张量。
    """
    def __init__(self,
                 excel_file_name="Transformer_FRA_indicator_Norm.xlsx",
                 data_sheet_name="FRA_indicator",
                 test_set_sheet_name="FRA_test_set_sample_pairs"):
        """
        初始化数据集，加载 Excel 文件中的数据。

        参数:
        - excel_file_name: Excel 文件名
        - data_sheet_name: 存放指标数据的 sheet 名称
        - test_set_sheet_name: 存放测试样本对的 sheet 名称
        """

        self.excel_file_name = excel_file_name
        self.data_sheet_name = data_sheet_name
        self.test_set_sheet_name = test_set_sheet_name

        # 加载数据
        First_sample_data, Second_sample_data, label = self.load_excel_csv(self.excel_file_name, self.data_sheet_name, self.test_set_sheet_name)

        # 转换为 PyTorch 张量
        # 每个样本 reshape 成 (1, 30)，方便后续模型输入
        self.First_sample_data = torch.tensor(First_sample_data, dtype=torch.float32).reshape(-1, 1, 30)
        self.Second_sample_data = torch.tensor(Second_sample_data, dtype=torch.float32).reshape(-1, 1, 30)
        self.label = torch.tensor(label, dtype=torch.long)

    def load_excel_csv(self, excel_file_name, data_sheet_name, test_set_sheet_name):
        """
        从 Excel 文件中加载数据。

        返回:
        - first_sample_data: 第一个样本数据 (numpy 数组)
        - second_sample_data: 第二个样本数据 (numpy 数组)
        - label: 标签 (numpy 数组)
        """

        # 指定需要的列
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
        # 读取 Excel 文件
        df_test_set = pd.read_excel(excel_file_name, sheet_name=test_set_sheet_name)
        df_data = pd.read_excel(excel_file_name, sheet_name=data_sheet_name)

        # 获取样本索引和标签
        First_class_rank = np.array(df_test_set['First_class_rank'])
        Second_class_rank = np.array(df_test_set['Second_class_rank'])
        Whether_similarity = np.array(df_test_set['Whether_similarity'])

        # 根据索引提取样本数据
        First_sample_data = np.array(df_data.iloc[First_class_rank][cols])
        Second_sample_data = np.array(df_data.iloc[Second_class_rank][cols])
        label = np.array(Whether_similarity)

        return First_sample_data, Second_sample_data, label

    def __len__(self):
        """返回数据集大小"""
        return self.First_sample_data.shape[0]

    def __getitem__(self, idx):
        """根据索引返回一个样本对及其标签"""
        return self.First_sample_data[idx], self.Second_sample_data[idx], self.label[idx]


def test_SFRA_indicator_dataset_test():
    dataset = SFRA_indicator_dataset_test()
    a = DataLoader(dataset, batch_size=32)
    for x1, x2, y in a:
        print(x1.shape, x2.shape, y.shape)
        print(y)

def test_SFRA_indicator_dataset_train():
    dataset = SFRA_indicator_dataset_train()
    a = DataLoader(dataset, batch_size=16, shuffle=True)
    for x in a:
        print(x["first_feat"].shape, x["first_fault_label"].shape, x["similarity"].shape)
        break


if __name__ == '__main__':
    test_SFRA_indicator_dataset_train()

