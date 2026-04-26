# -*- coding: utf-8 -*-
# Author: Yu Chen
# Data: 2026/2/28 21:55
# Email: yu2000.chen@connect.polyu.hk


import pandas as pd
import numpy as np
from pathlib import Path
import os
from FRA_indicators_utils import *
import random

def extract_FRA_data(path):
    FRA_df = pd.read_csv(
        str(path),
        encoding='utf-8',  # 或 'gbk' / 'utf-8-sig'（中文常见）
        sep=',',  # 分隔符（默认就是逗号，可改成 '\t' 等）
        header=0,  # 第几行是表头（0=第一行，None=无表头）
        skiprows=1,  # 跳过前2行（常用于跳过说明文字）
        nrows=1000,  # 只读前1000行（大文件调试用）
    )
    FRA = np.array(FRA_df)
    FRA_Frequency = FRA[:, 0]
    FRA_Amplitude = FRA[:, 1]
    FRA_Phase = FRA[:, 2]
    return FRA_Frequency, FRA_Amplitude, FRA_Phase

def sample_and_remove(lst, k=100):
    """从列表 lst 中随机抽取最多 k 个元素并在原列表中删除，返回抽取的元素列表。"""
    k = min(k, len(lst))
    if k == 0:
        return []
    indices = random.sample(range(len(lst)), k)
    sampled = [lst[i] for i in indices]
    for i in sorted(indices, reverse=True):
        lst.pop(i)
    return sampled

def build_transformer_fra_excel(
    transformer_fra_path="./变压器数据",
    excel_save_path="Transformer_FRA_indicator_Norm.xlsx",
    baseline_normal_fra_path="Normal_FRA_baseline.csv",
    test_number_per_class=100):
    # 读取基准（只读一次）
    _, fra_amplitude_baseline, _ = extract_FRA_data(baseline_normal_fra_path)
    TransformerFault_dict = {"AD": 0, "DSV": 1, "ITSC": 2, "RD": 3, "Normal": 4}

    # 收集所有行的列表（最后一次性转 DataFrame）
    rows = []
    class_list = []


    fault_type_list = os.listdir(transformer_fra_path)
    for fault_type in fault_type_list:
        file_path_temp_2 = os.path.join(transformer_fra_path, fault_type)
        fault_type_value = TransformerFault_dict[str(fault_type)]

        fault_file_name = os.listdir(file_path_temp_2)
        for file_name in fault_file_name:
            file_path_temp_3 = os.path.join(file_path_temp_2, file_name)
            fault_excel_name = os.listdir(file_path_temp_3)
            file_path_temp_4= os.path.join(file_path_temp_3, fault_excel_name[0])

            _, fra_amplitude, _ = extract_FRA_data(file_path_temp_4)

            # 计算所有指标(这里参考基准要放在前面)

            ssre_L, ssre_M, ssre_H = cal_ssre(fra_amplitude_baseline, fra_amplitude)
            sd_L, sd_M, sd_H = cal_sd(fra_amplitude_baseline, fra_amplitude)
            rou_L, rou_M, rou_H = cal_rou(fra_amplitude_baseline, fra_amplitude)
            mm_L, mm_M, mm_H = cal_mm(fra_amplitude_baseline, fra_amplitude)
            ed_L, ed_M, ed_H = cal_ed(fra_amplitude_baseline, fra_amplitude)
            e_L, e_M, e_H = cal_e(fra_amplitude_baseline, fra_amplitude)
            delta_L, delta_M, delta_H = cal_delta(fra_amplitude_baseline, fra_amplitude)
            dabs_L, dabs_M, dabs_H = cal_dabs(fra_amplitude_baseline, fra_amplitude)
            cc_L, cc_M, cc_H = cal_cc(fra_amplitude_baseline, fra_amplitude)
            asle_L, asle_M, asle_H = cal_asle(fra_amplitude_baseline, fra_amplitude)

            row = {
                "fault_type": fault_type,
                "fault_type_code": fault_type_value,
                "ssre_l": ssre_L, "ssre_m": ssre_M, "ssre_h": ssre_H,
                "sd_l": sd_L, "sd_m": sd_M, "sd_h": sd_H,
                "rou_l": rou_L, "rou_m": rou_M, "rou_h": rou_H,
                "mm_l": mm_L, "mm_m": mm_M, "mm_h": mm_H,
                "ed_l": ed_L, "ed_m": ed_M, "ed_h": ed_H,
                "e_l": e_L, "e_m": e_M, "e_h": e_H,
                "delta_l": delta_L, "delta_m": delta_M, "delta_h": delta_H,
                "dabs_l": dabs_L, "dabs_m": dabs_M, "dabs_h": dabs_H,
                "cc_l": cc_L, "cc_m": cc_M, "cc_h": cc_H,
                "asle_l": asle_L, "asle_m": asle_M, "asle_h": asle_H,
            }
            rows.append(row)
            class_list.append(fault_type)

    # 一次性生成 DataFrame
    df = pd.DataFrame(rows)

    # 仅对指定列做 Min-Max 归一化
    cols_to_normalize = [
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

    existing = [c for c in cols_to_normalize if c in df.columns]
    if existing:
        # 进一步只保留数值列
        numeric_cols = df[existing].select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            mins = df[numeric_cols].min()
            maxs = df[numeric_cols].max()
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0  # 常数列避免除以零，归一化后为0
            df[numeric_cols] = (df[numeric_cols] - mins) / ranges

    # 生成pair对
    positive_sample_list = []
    negative_sample_list = []

    class_1_rank = 0
    for class_1 in class_list:
        class_2_rank = 0
        for class_2 in class_list:
            if class_1 == class_2:
                positive_sample_list.append({"First_class":class_1,
                                             "First_class_rank": class_1_rank,
                                             "Second_class": class_2,
                                             "Second_class_rank": class_2_rank,
                                             "Whether_similarity": 1})
            else:
                negative_sample_list.append({"First_class":class_1,
                                             "First_class_rank": class_1_rank,
                                             "Second_class": class_2,
                                             "Second_class_rank": class_2_rank,
                                             "Whether_similarity": 0})

            class_2_rank += 1
        class_1_rank += 1

    # 随机从正样本和负样本中各抽出100个作为测试集并在原列表中删除
    test_set_sample_list = []
    sampled_pos = sample_and_remove(positive_sample_list, test_number_per_class)
    sampled_neg = sample_and_remove(negative_sample_list, test_number_per_class)

    # 转为 DataFrame 并合并为一个测试集 DataFrame
    df_sampled_pos = pd.DataFrame(sampled_pos)
    df_sampled_neg = pd.DataFrame(sampled_neg)

    # 合并并重置索引，必要时随机打乱（保证训练/测试顺序无偏）
    df_test_set_sample = pd.concat([df_sampled_pos, df_sampled_neg], ignore_index=True)
    df_test_set_sample = df_test_set_sample.sample(frac=1, random_state=42).reset_index(drop=True)

    df_positive_sample = pd.DataFrame(positive_sample_list)
    df_negative_sample = pd.DataFrame(negative_sample_list)


    # 正确使用 writer 写入多个 sheet
    with pd.ExcelWriter(excel_save_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="FRA_indicator", index=False)
        df_positive_sample.to_excel(writer, sheet_name="FRA_positive_sample_pairs", index=False)
        df_negative_sample.to_excel(writer, sheet_name="FRA_negative_sample_pairs", index=False)
        df_test_set_sample.to_excel(writer, sheet_name="FRA_test_set_sample_pairs", index=False)
    print(
        f"已保存 {len(df)} 条记录，训练正样本 {len(df_positive_sample)}，训练负样本 {len(df_negative_sample)} 到 → `excel_save_path`")


if __name__ == '__main__':
    build_transformer_fra_excel()