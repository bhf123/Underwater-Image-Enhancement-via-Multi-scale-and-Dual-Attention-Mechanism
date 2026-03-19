import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models


# --- Perceptual loss network  --- #
# class LossNetwork(torch.nn.Module):
#     def __init__(self, vgg_model):
#         super(LossNetwork, self).__init__()
#         self.vgg_layers = vgg_model
#         self.layer_name_mapping = {
#             '3': "relu1_2",
#             '8': "relu2_2",
#             '15': "relu3_3"
#         }
#
#     def output_features(self, x):
#         output = {}
#         for name, module in self.vgg_layers._modules.items():
#             x = module(x)
#             if name in self.layer_name_mapping:
#                 output[self.layer_name_mapping[name]] = x
#         return list(output.values())
#
#     def forward(self, dehaze, gt):
#         loss = []
#         dehaze_features = self.output_features(dehaze)
#         gt_features = self.output_features(gt)
#         for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
#             loss.append(F.mse_loss(dehaze_feature, gt_feature))
#
#         return sum(loss)/len(loss)

class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '1': "relu1_1",  # 更改为 '1'，表示第一个卷积层的输出
            '6': "relu2_2",
            '11': "relu3_3",
            '20': "relu4_3",  # 添加一个更深层级的表示
            # '25': "relu5_3"  # 添加更多层级
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss) / len(loss)