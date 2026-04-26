# -*- coding: utf-8 -*-
# Author: Yu Chen
# Data: 2026/3/1 19:47
# Email: yu2000.chen@connect.polyu.hk


import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, features_d=5, features_size=30):
        super(CNNFeatureExtractor, self).__init__()
        self.features_size = features_size
        self.features_d = features_d
        self.in_channels = in_channels

        # model architecture
        self.layer1 = nn.Conv1d(in_channels, features_d, kernel_size=1, stride=1, padding=1)
        self.layer2 = self._block(features_d, features_d * 2, 4, 1, 0)
        self.layer3 = self._block(features_d * 2, features_d * 4, 4, 1, 0)
        self.layer4 = self._block(features_d * 4, features_d * 8, 4, 1, 0)
        self.layer5 = nn.Conv1d(features_d * 8, 1, kernel_size=4, stride=2, padding=0)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm1d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        y = self.layer5(x)
        return y


class LSTMFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, features_d=5, features_size=30):
        super(LSTMFeatureExtractor, self).__init__()
        self.features_size = features_size
        self.features_d = features_d
        self.in_channels = in_channels

        # 最优配置：单层双向LSTM，hidden=features_d，双向输出正好10维
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=features_d,
            num_layers=5,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        # x: [batch, 1, 30] → [batch, 30, 1]
        if x.dim() < 3:
            x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)  # out: [batch, 30, 10]
        y = out[:, -1, :].unsqueeze(1)  # 取最后时间步 → [batch, 1, 10]
        return y


class RNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, features_d=5, features_size=30):
        super(RNNFeatureExtractor, self).__init__()
        self.features_size = features_size
        self.features_d = features_d
        self.in_channels = in_channels

        self.rnn = nn.RNN(
            input_size=in_channels,
            hidden_size=features_d,
            num_layers=5,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        if x.dim() < 3:
            x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        y = out[:, -1, :].unsqueeze(1)
        return y


class GRUFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, features_d=5, features_size=30):
        super(GRUFeatureExtractor, self).__init__()
        self.features_size = features_size
        self.features_d = features_d
        self.in_channels = in_channels

        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=features_d,
            num_layers=5,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        if x.dim() < 3:
            x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        y = out[:, -1, :].unsqueeze(1)
        return y


class NNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, features_d=5, features_size=30):
        super(NNFeatureExtractor, self).__init__()
        self.features_size = features_size
        self.features_d = features_d
        self.in_channels = in_channels

        input_dim = features_size * in_channels  # 30

        # 完全镜像CNN的通道增长逻辑：30 → features_d → *2 → *4 → *8 → 10
        self.fc = nn.Sequential(
            nn.Linear(input_dim, features_d),
            nn.LeakyReLU(0.2),
            nn.Linear(features_d, features_d * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(features_d * 2, features_d * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(features_d * 4, features_d * 8),
            nn.LeakyReLU(0.2),
            nn.Linear(features_d * 8, 10),
        )

    def forward(self, x):
        # x: [batch, 1, 30] → [batch, 30]
        x = x.view(x.size(0), -1)
        y = self.fc(x)  # [batch, 10]
        y = y.unsqueeze(1)  # [batch, 1, 10]
        return y


class TransformerEncoderFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, features_d=5, features_size=30):
        super(TransformerEncoderFeatureExtractor, self).__init__()
        self.features_size = features_size
        self.features_d = features_d
        self.in_channels = in_channels
        self.d_model = features_d * 2  # 10，与 RNN/LSTM/GRU 双向输出一致

        # 1×1 投影：把最后一个维度从 1 变成 d_model=10
        self.input_proj = nn.Linear(in_channels, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=2,
            dim_feedforward=self.d_model * 4,  # 40
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=False,  # Pre-LN 更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=5)

    def forward(self, x):
        if x.dim() < 3:
            x = x.unsqueeze(1)
        # 例如：x = x.permute(0,2,1) 后接 nn.Linear(1,10) 或 x.repeat(1,1,10) 等
        x = x.permute(0, 2, 1)
        # 投影到 d_model 维度
        x = self.input_proj(x)  # → [batch, 30, 10]
        x = self.transformer(x)  # [batch, 30, 10]
        y = x.mean(dim=1).unsqueeze(1)  # Global Average Pooling → [batch, 1, 10]
        return y


# ====== Siamese 网络 ======
class SiameseNetwork(nn.Module):
    def __init__(self, model_name="CNN"):
        super(SiameseNetwork, self).__init__()
        if model_name == "CNN":
            self.feature_extractor = CNNFeatureExtractor()
        elif model_name == "LSTM":
            self.feature_extractor = LSTMFeatureExtractor()
        elif model_name == "RNN":
            self.feature_extractor = RNNFeatureExtractor()
        elif model_name == "GRU":
            self.feature_extractor = GRUFeatureExtractor()
        elif model_name == "NN":
            self.feature_extractor = NNFeatureExtractor()
        elif model_name == "Transformer":
            self.feature_extractor = TransformerEncoderFeatureExtractor()

        self.fc_out = nn.Linear(10*2, 1)

    def forward(self, FRA_indicator_1, FRA_indicator_2):
        f1 = self.feature_extractor(FRA_indicator_1)
        f2 = self.feature_extractor(FRA_indicator_2)
        f1 = f1.squeeze(1)
        f2 = f2.squeeze(1)
        #combined = torch.cat((f1, f2), dim=1)
        out = torch.cosine_similarity(f1, f2)
        #out = torch.sigmoid(self.fc_out(combined))
        out = out.unsqueeze(1)  # [batch, 1]
        return out, f1, f2


if __name__ == '__main__':
    x = torch.randn(1, 1, 30)
    CNN_model = CNNFeatureExtractor()
    LSTM_model = LSTMFeatureExtractor()
    RNN_model = RNNFeatureExtractor()
    GRU_model = GRUFeatureExtractor()
    NN_model = NNFeatureExtractor()
    TransformerEncoder_model = TransformerEncoderFeatureExtractor()
    SiameseNetwork_model = SiameseNetwork(model_name="GRU")
    out_params = sum(p.numel() for p in SiameseNetwork_model.parameters())
    print(f"参数总数: {out_params}")

    #out, f1, f2 = SiameseNetwork_model(x, x)
    #print(out.shape)

