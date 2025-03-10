import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        # 加载预训练的MobileNetV2并冻结参数
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.feature_extractor = self.mobilenet.features
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(1280 + 1, 256, kernel_size=1),  # 1280来自MobileNetV2 + 1来自LBP
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, lbp):
        # 从MobileNetV2提取深度特征
        deep_features = self.feature_extractor(x)  # 形状: B x 1280 x H/32 x W/32
        
        # 将LBP纹理图与深度特征尺寸对齐
        lbp_resized = F.interpolate(lbp, size=deep_features.shape[2:], 
                                    mode='bilinear', align_corners=False)
        
        # 特征拼接
        concat_features = torch.cat([deep_features, lbp_resized], dim=1)
        
        # 生成注意力图
        attention_map = self.fusion(concat_features)
        
        # 调整回原始分辨率
        attention_map = F.interpolate(attention_map, size=(x.shape[2], x.shape[3]), 
                                     mode='bilinear', align_corners=False)
        
        return attention_map


class ChannelAttentionModule(nn.Module):
    def __init__(self, init_y=0.3, init_u=0.5, init_v=0.5):
        super(ChannelAttentionModule, self).__init__()
        # YUV通道的可学习权重
        self.y_weight = nn.Parameter(torch.tensor(init_y))
        self.u_weight = nn.Parameter(torch.tensor(init_u))
        self.v_weight = nn.Parameter(torch.tensor(init_v))
        
    def forward(self, yuv_frame):
        # YUV帧应为形状为[B, 3, H, W]的张量
        # 其中通道0是Y，1是U，2是V
        y = yuv_frame[:, 0:1, :, :] * self.y_weight
        u = yuv_frame[:, 1:2, :, :] * self.u_weight
        v = yuv_frame[:, 2:3, :, :] * self.v_weight
        
        # 连接加权通道
        weighted_yuv = torch.cat([y, u, v], dim=1)
        
        return weighted_yuv