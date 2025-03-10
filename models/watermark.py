import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.feature import local_binary_pattern
import cv2

from models.attention import SpatialAttentionModule, ChannelAttentionModule
from models.dct import dct_embed, dct_extract
from models.geometric import sift_correction

class TextureDrivenWatermark(nn.Module):
    def __init__(self):
        super(TextureDrivenWatermark, self).__init__()
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule()
        
    def rgb_to_yuv(self, rgb):
        """将RGB张量转换为YUV"""
        # 使用BT.601标准进行RGB->YUV转换
        y = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        u = -0.147 * rgb[:, 0:1] - 0.289 * rgb[:, 1:2] + 0.436 * rgb[:, 2:3]
        v = 0.615 * rgb[:, 0:1] - 0.515 * rgb[:, 1:2] - 0.100 * rgb[:, 2:3]
        return torch.cat([y, u, v], dim=1)
    
    def yuv_to_rgb(self, yuv):
        """将YUV张量转换为RGB"""
        # 使用BT.601标准进行YUV->RGB转换
        r = yuv[:, 0:1] + 1.140 * yuv[:, 2:3]
        g = yuv[:, 0:1] - 0.395 * yuv[:, 1:2] - 0.581 * yuv[:, 2:3]
        b = yuv[:, 0:1] + 2.032 * yuv[:, 1:2]
        return torch.cat([r, g, b], dim=1)
    
    def compute_lbp(self, rgb_tensor):
        """从RGB张量计算LBP特征"""
        # 将张量转换为numpy以进行LBP计算
        batch_size = rgb_tensor.size(0)
        lbp_batch = []
        
        for i in range(batch_size):
            # 转换为灰度和numpy
            gray = 0.299 * rgb_tensor[i, 0] + 0.587 * rgb_tensor[i, 1] + 0.114 * rgb_tensor[i, 2]
            gray_np = gray.detach().cpu().numpy()
            
            # 计算LBP（半径=1，8个点）
            lbp = local_binary_pattern(gray_np, 8, 1, method='uniform')
            
            # 归一化到[0, 1]
            lbp = lbp / lbp.max()
            
            # 转换回张量
            lbp_tensor = torch.from_numpy(lbp).float().unsqueeze(0).to(rgb_tensor.device)
            lbp_batch.append(lbp_tensor)
        
        return torch.stack(lbp_batch)
    
    def embed_watermark(self, frame, watermark, texture_mask=None):
        """
        使用纹理驱动方法在帧中嵌入水印
        
        参数:
            frame: 输入帧（RGB张量）
            watermark: 二进制水印位
            texture_mask: 可选的预先计算的纹理掩码
            
        返回:
            嵌入水印的帧（RGB张量）
        """
        # 将帧转换为YUV
        yuv_frame = self.rgb_to_yuv(frame)
        
        # 应用通道注意力
        weighted_yuv = self.channel_attention(yuv_frame)
        
        # 如果未提供纹理掩码，则计算LBP特征
        if texture_mask is None:
            # 计算用于纹理分析的LBP
            lbp_tensor = self.compute_lbp(frame)
            
            # 计算基于纹理的空间注意力
            texture_mask = self.spatial_attention(frame, lbp_tensor)
        
        # 将张量转换为numpy以进行DCT嵌入
        y_channel = weighted_yuv[:, 0].detach().cpu().numpy()
        texture_mask_np = texture_mask.squeeze(1).detach().cpu().numpy()
        watermark_np = np.array(watermark)
        
        # 在批次中的每个图像中嵌入水印
        watermarked_y = []
        for i in range(y_channel.shape[0]):
            # 使用DCT在Y通道中嵌入水印
            marked_y = dct_embed(y_channel[i], watermark_np, texture_mask_np[i])
            watermarked_y.append(torch.from_numpy(marked_y).float())
        
        # 将批次张量堆叠回去
        watermarked_y = torch.stack(watermarked_y).to(frame.device)
        
        # 与原始UV通道组合
        watermarked_yuv = torch.cat([watermarked_y.unsqueeze(1), 
                                     weighted_yuv[:, 1:2], 
                                     weighted_yuv[:, 2:3]], dim=1)
        
        # 转换回RGB
        watermarked_rgb = self.yuv_to_rgb(watermarked_yuv)
        
        return watermarked_rgb, texture_mask
    
    def extract_watermark(self, watermarked_frame, watermark_length, apply_correction=False, reference_frame=None):
        """
        从嵌入水印的帧中提取水印
        
        参数:
            watermarked_frame: 嵌入水印的帧（RGB张量）
            watermark_length: 原始水印的长度
            apply_correction: 是否应用几何校正
            reference_frame: 用于几何校正的参考帧
            
        返回:
            提取的水印位
        """
        # 如果需要几何校正且有参考可用
        if apply_correction and reference_frame is not None:
            # 将张量转换为numpy以进行SIFT
            watermarked_np = watermarked_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            reference_np = reference_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            
            # 转换为灰度以进行SIFT处理
            watermarked_gray = cv2.cvtColor(watermarked_np, cv2.COLOR_RGB2GRAY)
            reference_gray = cv2.cvtColor(reference_np, cv2.COLOR_RGB2GRAY)
            
            # 应用几何校正
            corrected_gray = sift_correction(reference_gray, watermarked_gray)
            
            # 转换回张量
            corrected_tensor = torch.from_numpy(corrected_gray).float().unsqueeze(0).unsqueeze(0).to(watermarked_frame.device)
            
            # 从校正后的帧提取
            y_channel = corrected_tensor.squeeze().detach().cpu().numpy()
        else:
            # 转换为YUV
            yuv_frame = self.rgb_to_yuv(watermarked_frame)
            
            # 从Y通道提取
            y_channel = yuv_frame[:, 0].detach().cpu().numpy()
        
        # 从批次中的每个图像提取水印
        extracted_watermarks = []
        for i in range(y_channel.shape[0]):
            extracted_watermark = dct_extract(y_channel[i], watermark_length)
            extracted_watermarks.append(extracted_watermark)
        
        return extracted_watermarks
    
    def forward(self, frame, watermark):
        """用于训练的前向传递"""
        # 计算纹理特征
        lbp_tensor = self.compute_lbp(frame)
        
        # 计算基于纹理的空间注意力
        texture_mask = self.spatial_attention(frame, lbp_tensor)
        
        # 嵌入水印
        watermarked_frame, _ = self.embed_watermark(frame, watermark, texture_mask)
        
        return watermarked_frame, texture_mask