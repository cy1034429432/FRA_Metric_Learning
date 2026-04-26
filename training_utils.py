# -*- coding: utf-8 -*-
# Author: Yu Chen
# Data: 2026/3/3 09:46
# Email: yu2000.chen@connect.polyu.hk
import pandas as pd

from model_utils import SiameseNetwork
from dataset_utils import SFRA_indicator_dataset_train, SFRA_indicator_dataset_test
from FRA_indicator_calculation import build_transformer_fra_excel
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)


def set_seed(seed: int = 3407):
    """
    固定随机种子，尽可能让实验可复现

    参数:
        seed: 随机种子值（建议使用 0~2^32-1 之间的整数）

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 当前默认 GPU
        torch.cuda.manual_seed_all(seed)  # 所有 GPU


def vat_loss(model, feat, xi=10.0, eps=0.01, ip=3):
    """
    vat_loss 实现了 Virtual Adversarial Training (VAT) 的损失函数。它的目标是通过在输入特征上添加小的对抗扰动，来提高模型对输入分布的鲁棒性。具体流程是：

    1. 获取模型在原始输入上的预测分布。
    2. 初始化一个随机扰动向量。
    3. 迭代更新扰动方向，使其最大化预测分布的变化。
    4. 使用最终扰动生成对抗样本，并计算原始预测分布与对抗预测分布之间的 KL 散度作为损失。
    """
    # 定义 KL 散度损失函数，用于衡量扰动前后预测分布的差异
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    with torch.no_grad():
        # 得到模型在原始输入上的预测结果
        _, pred, _ = model(feat, feat)
        # 将预测结果转为概率分布（softmax）
        pred = F.softmax(pred, dim=1)

    # 初始化扰动
    d = torch.randn_like(feat)

    # 迭代寻找最优扰动方向
    for _ in range(ip):
        # 将 d 归一化并放大到 xi 的尺度
        d = xi * F.normalize(d, p=2, dim=1)
        # 设置 d 为可导，以便后续计算梯度
        d.requires_grad_()
        # 在扰动后的输入上运行模型
        _, pred_hat, _ = model(feat + d, feat)
        # 使用模型原始预测的 argmax 作为伪标签，计算交叉熵损失
        adv_loss = F.cross_entropy(pred_hat, pred.argmax(dim=1))
        # 计算 adv_loss 对 d 的梯度
        grad = torch.autograd.grad(adv_loss, d, retain_graph=True)[0]
        # 用梯度更新扰动方向，并断开计算图
        d = grad.detach()
        # 再次归一化梯度方向
        d = F.normalize(d, p=2, dim=1)

    # 得到最终的对抗扰动 r_adv
    r_adv = eps * d
    # 在加上最终扰动的输入上运行模型
    _, pred_hat, _ = model(feat + r_adv, feat)

    # 将预测结果转为 log 概率分布
    pred_hat = F.log_softmax(pred_hat, dim=1)
    # 计算扰动前后分布的 KL 散度
    return kl_loss(pred_hat, pred)


def calculate_r_adv(model, feat, xi=10.0, eps=0.005, ip=3):

    with torch.no_grad():
        # 得到模型在原始输入上的预测结果
        _, pred, _ = model(feat, feat)
        # 将预测结果转为概率分布（softmax）
        pred = F.softmax(pred, dim=1)

    # 初始化扰动
    d = torch.randn_like(feat)

    # 迭代寻找最优扰动方向
    for _ in range(ip):
        # 将 d 归一化并放大到 xi 的尺度
        d = xi * F.normalize(d, p=2, dim=1)
        # 设置 d 为可导，以便后续计算梯度
        d.requires_grad_()
        # 在扰动后的输入上运行模型
        _, pred_hat, _ = model(feat + d, feat)
        # 使用模型原始预测的 argmax 作为伪标签，计算交叉熵损失
        adv_loss = F.cross_entropy(pred_hat, pred.argmax(dim=1))
        # 计算 adv_loss 对 d 的梯度
        grad = torch.autograd.grad(adv_loss, d, retain_graph=True)[0]
        # 用梯度更新扰动方向，并断开计算图
        d = grad.detach()
        # 再次归一化梯度方向
        d = F.normalize(d, p=2, dim=1)

    # 得到最终的对抗扰动 r_adv
    r_adv = eps * d

    return r_adv



def total_loss(model, output, similarity,
               first_feat, f1, f1_pos_first, f2_neg_first,
               second_feat, f2, f1_pos_second, f2_neg_second,
               triplet_loss_c=0.3, vaT_loss_c=0.1):
    """
    Function 描述：
    total_loss 用于综合不同的损失函数，以训练模型时同时考虑多方面的约束。
    它包含三类损失：
    1. BCE/MSE 损失：衡量模型输出与相似度标签之间的差异。
    2. Triplet 损失：通过正样本和负样本约束特征空间，使相似样本更接近，不相似样本更远。
    3. VAT 损失：通过虚拟对抗训练增强模型的鲁棒性，使模型在输入扰动下保持预测稳定。

    参数说明：
    - model: 神经网络模型
    - output: 模型的预测输出
    - similarity: 样本之间的相似度标签
    - first_feat, second_feat: 输入特征，用于 VAT loss
    - f1, f2: anchor 特征
    - f1_pos_first, f1_pos_second: 正样本特征
    - f2_neg_first, f2_neg_second: 负样本特征
    - triplet_loss_c: triplet 损失的权重系数
    - vaT_loss_c: VAT 损失的权重系数

    返回值：
    - 综合损失值 (bce_loss + triplet_loss + vat_loss)
    """

    # 将 similarity 扩展维度，保证与 output 对齐
    similarity = similarity.unsqueeze(1)

    # 使用 MSELoss 计算预测输出与相似度标签之间的差异
    bce_loss = nn.MSELoss()(output, similarity.float())

    # Triplet 损失：约束 f1 与正样本接近，f1 与负样本远离
    triplet_loss_1 = triplet_loss_fn(f1, f1_pos_first, f2_neg_first)
    triplet_loss_2 = triplet_loss_fn(f2, f1_pos_second, f2_neg_second)

    # VAT 损失：增强模型对输入扰动的鲁棒性
    vat_loss_1 = vat_loss(model, first_feat)
    vat_loss_2 = vat_loss(model, second_feat)

    total_triplet_loss = triplet_loss_c * (triplet_loss_1 + triplet_loss_2)
    total_vat_loss = vaT_loss_c * (vat_loss_1 + vat_loss_2)

    # 返回综合损失：MSE + Triplet + VAT
    return (bce_loss + total_triplet_loss  + total_vat_loss,
            bce_loss.item(),
            total_triplet_loss.item(),
            total_vat_loss.item())



def train(model, train_loader,  test_loader, optimizer, device,
        logs_dir, epochs=100,
        whether_enhance_data=True,
        excel_file_name="Transformer_FRA_indicator_Norm.xlsx",
        data_sheet_name="FRA_indicator"):

    write = SummaryWriter(logs_dir)
    best_loss = float('inf')  # 初始为正无穷
    best_epoch = -1
    save_path = f"{logs_dir}/best_model.pth"  # 你可以改成带时间戳或 epoch 的名字
    # t-sne
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    for epoch in range(epochs):

        total_epoch_loss = 0
        total_bce_loss = 0
        total_triplet_loss = 0
        total_vat_loss = 0

        for batch_idx, sample_dict in enumerate(train_loader):
            model.train()

            first_feat = sample_dict["first_feat"].to(device)
            first_fault_label = sample_dict["first_fault_label"]
            second_feat = sample_dict["second_feat"].to(device)
            second_fault_label = sample_dict["second_fault_label"]
            similarity = sample_dict["similarity"].to(device)
            pos_feat_first = sample_dict["pos_feat_first"].to(device)
            pos_fault_label_first = sample_dict["pos_fault_label_first"]
            neg_feat_first = sample_dict["neg_feat_first"].to(device)
            neg_fault_label_first = sample_dict["neg_fault_label_first"]
            pos_feat_second = sample_dict["pos_feat_second"].to(device)
            pos_fault_label_second = sample_dict["pos_fault_label_second"]
            neg_feat_second = sample_dict["neg_feat_second"].to(device)
            neg_fault_label_second = sample_dict["neg_fault_label_second"]

            if whether_enhance_data == True:
                # 统计正负样本数量
                positive_sample_number = (similarity == 1).sum().item()
                negative_sample_number = (similarity == 0).sum().item()

                if negative_sample_number > positive_sample_number:
                    vat_make_sample_number = negative_sample_number - positive_sample_number

                    # 通过 VAT 制造等额的正样本
                    base_feat_samples = []
                    vat_samples = []
                    vat_labels = []
                    vat_pos_samples = []
                    vat_neg_samples = []

                    # 随机选择一些正样本作为基底
                    pos_indices = (similarity == 1).nonzero(as_tuple=True)[0]

                    for i in range(vat_make_sample_number):

                        idx = random.randint(0, first_feat.shape[0]-1)  # 循环取正样本
                        base_feat = first_feat[idx]  # 取出一个正样本特征

                        # 计算对抗扰动并生成新的正样本
                        r_adv = calculate_r_adv(model, base_feat)
                        vat_feat = base_feat + r_adv

                        base_feat_samples.append(base_feat)
                        vat_samples.append(vat_feat)
                        vat_labels.append(torch.tensor([1.0], device=device))
                        vat_pos_samples.append(pos_feat_first[idx])
                        vat_neg_samples.append(neg_feat_first[idx])  # 正样本标签

                    # 拼接到原始 batch 中
                    base_feat_samples = torch.cat(base_feat_samples, dim=0).unsqueeze(1)
                    vat_samples = torch.cat(vat_samples, dim=0).unsqueeze(1)
                    vat_labels = torch.cat(vat_labels, dim=0)
                    vat_pos_samples = torch.cat(vat_pos_samples, dim=0).unsqueeze(1)
                    vat_neg_samples = torch.cat(vat_neg_samples, dim=0).unsqueeze(1)

                    first_feat = torch.cat([first_feat, base_feat_samples], dim=0)
                    second_feat = torch.cat([second_feat, vat_samples], dim=0)
                    similarity = torch.cat([similarity, vat_labels], dim=0)
                    pos_feat_first = torch.cat([pos_feat_first, vat_pos_samples], dim=0)
                    neg_feat_first = torch.cat([neg_feat_first, vat_neg_samples], dim=0)
                    pos_feat_second = torch.cat([pos_feat_second, vat_pos_samples], dim=0)
                    neg_feat_second = torch.cat([neg_feat_second, vat_neg_samples], dim=0)


            similarity = similarity.to(dtype=torch.float32)

            optimizer.zero_grad()
            output, f1, f2 = model(first_feat, second_feat)
            _, f1_pos_first, f2_neg_first = model(pos_feat_first, neg_feat_first)
            _, f1_pos_second, f2_neg_second = model(pos_feat_second, neg_feat_second)

            (loss, bce_loss, total_triplet_loss, total_vat_loss) = total_loss(model, output, similarity,
                              first_feat, f1, f1_pos_first, f2_neg_first,
                              second_feat, f2, f1_pos_second, f2_neg_second)

            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            total_bce_loss += bce_loss
            total_triplet_loss += total_triplet_loss
            total_vat_loss += total_vat_loss

        # 保存模型（推荐保存 state_dict）
        if total_epoch_loss/len(train_loader) < best_loss:
            best_loss = total_epoch_loss/len(train_loader)
            best_epoch = epoch + 1
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                # 可选：'scheduler_state_dict': scheduler.state_dict() 如果有 scheduler
            }, save_path)


        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_epoch_loss / len(train_loader):.4f}")

        metrics = evaluate(model, test_loader, device)
        # record data
        write.add_scalar("training_set/total_loss", total_epoch_loss/len(train_loader), epoch+1)
        write.add_scalar("training_set/bce_loss", total_bce_loss/len(train_loader), epoch+1)
        write.add_scalar("training_set/triplet_loss", total_triplet_loss/len(train_loader), epoch+1)
        write.add_scalar("training_set/vat_loss", total_vat_loss/len(train_loader), epoch+1)
        write.add_scalar("test_set/accuracy", metrics["accuracy"], epoch+1)
        write.add_scalar("test_set/precision", metrics["precision"], epoch+1)
        write.add_scalar("test_set/recall", metrics["recall"], epoch+1)
        write.add_scalar("test_set/f1", metrics["f1"], epoch+1)
        write.add_scalar("test_set/auc", metrics["auc"], epoch+1)

        model.eval()
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

            fig_tsne = plt.figure(figsize=(10, 8), dpi=120)
            # plt.scatter(all_sample_feature_tsne[df_data["fault_type_code"] == 0][0], all_sample_feature_tsne[df_data["fault_type_code"] == 0][1], c="r", marker="*")
            # plt.scatter(all_sample_feature_tsne[df_data["fault_type_code"] == 1][0], all_sample_feature_tsne[df_data["fault_type_code"] == 1][1], c="r", marker="*")
            # plt.scatter(all_sample_feature_tsne[df_data["fault_type_code"] == 2][0], all_sample_feature_tsne[df_data["fault_type_code"] == 2][1], c="r", marker="*")
            # plt.scatter(all_sample_feature_tsne[df_data["fault_type_code"] == 3][0], all_sample_feature_tsne[df_data["fault_type_code"] == 3][1], c="r", marker="*")
            # plt.scatter(all_sample_feature_tsne[df_data["fault_type_code"] == 4][0], all_sample_feature_tsne[df_data["fault_type_code"] == 4][1], c="r", marker="*")

            fault_types = ["AD", "DSV", "ITSC", "RD", "Normal"]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']  # 柔和且区分度高的颜色
            markers = ['o', 's', '^', 'D', '*']  # 不同形状辅助区分
            alpha = 0.7  # 轻微透明，避免重叠太严重
            s = 60  # 点的大小

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
                    label=f"{fault_types[code]} ({code})"
                )

            # 美化设置
            plt.title("t-SNE Visualization of Transformer Fault Features", fontsize=16, pad=15)
            plt.xlabel("t-SNE Dimension 1", fontsize=12)
            plt.ylabel("t-SNE Dimension 2", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(
                title="Fault Type",
                loc='best',  # 或 'upper right', 'lower left' 等
                fontsize=11,
                title_fontsize=13,
                frameon=True,
                shadow=True,
                fancybox=True
            )
            write.add_figure(f"test_set/t-sne", figure=fig_tsne, global_step=(epoch + 1))
    write.close()





def evaluate(model, test_loader, device, threshold=0.5):
    """
    评估模型在测试集上的多种指标
    假设：
    - model 输出是经过 sigmoid 的相似度分数 (0~1之间)
    - 阈值 threshold 以上判为 1（相似），否则为 0
    """
    model.eval()

    all_preds = []  # 二值预测 (0/1)
    all_probs = []  # 原始概率 (用于 AUC)
    all_labels = []  # 真实标签

    with torch.no_grad():
        for first, second, labels in test_loader:
            first = first.to(device)
            second = second.to(device)
            labels = labels.to(device)

            output, _, _ = model(first, second)  # 预期 shape: (batch_size,) 或 (batch_size, 1)

            # 确保是 1D
            if output.dim() > 1:
                output = output.squeeze(-1)

            # 收集概率和标签（用于后续计算）
            probs = output.cpu().numpy()  # shape: (batch_size,)
            binary_preds = (output > threshold).long().cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(binary_preds)
            all_labels.extend(labels.cpu().numpy())

    # 转换为 numpy 数组，便于 sklearn 计算
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 计算各种指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # AUC 需要概率值，且标签必须至少包含 0 和 1
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')  # 如果全是同一类标签，auc 无法计算

    # 打印结果
    print("Evaluation on Test Set:")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")

    # 返回一个字典，方便后续使用
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

    return metrics

def main():

    seed_number = 3407
    learning_rate = 0.0001
    whether_enhance_data = False
    resample_strategy = "oversample"  # 'none' | 'oversample' | 'undersample'
    model_type = "CNN"
    logs_dir = f"./logs/{model_type}_{resample_strategy}_enhance_{whether_enhance_data}"
    whether_need_build_dataset = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed_number)
    model = SiameseNetwork(model_name=model_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 数据集准备
    if whether_need_build_dataset== True:
        build_transformer_fra_excel(
            transformer_fra_path="./变压器数据",
            excel_save_path="Transformer_FRA_indicator_Norm.xlsx",
            baseline_normal_fra_path="Normal_FRA_baseline.csv",
            test_number_per_class=100)
    train_dataset = SFRA_indicator_dataset_train(excel_file_name = "Transformer_FRA_indicator_Norm.xlsx",
                                                 data_sheet_name = "FRA_indicator",
                                                 positive_sample_sheet_name = "FRA_positive_sample_pairs",
                                                 negative_sample_sheet_name = "FRA_negative_sample_pairs",
                                                 resample = resample_strategy,  # 'none' | 'oversample' | 'undersample'
                                                 random_state = 3407)

    test_dataset = SFRA_indicator_dataset_test(
                 excel_file_name="Transformer_FRA_indicator_Norm.xlsx",
                 data_sheet_name="FRA_indicator",
                 test_set_sheet_name="FRA_test_set_sample_pairs")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 训练和测试
    train(model, train_loader, test_loader, optimizer, device,
          logs_dir=logs_dir, epochs=100, whether_enhance_data=whether_enhance_data,
          excel_file_name = "Transformer_FRA_indicator_Norm.xlsx",
          data_sheet_name = "FRA_indicator")


if __name__ == '__main__':
    main()

