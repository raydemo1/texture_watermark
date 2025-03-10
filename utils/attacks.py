import torch
import numpy as np
import cv2
from PIL import Image
import io

def apply_attack(frames, attack_type):
    """
    对嵌入水印的帧应用各种攻击
    
    参数:
        frames: 嵌入水印的帧张量
        attack_type: 要应用的攻击类型
        
    返回:
        受攻击的帧张量
    """
    device = frames.device
    attacked_frames = []
    
    for i in range(frames.size(0)):
        # 转换为numpy以进行攻击
        frame = frames[i].detach().cpu().numpy().transpose(1, 2, 0)
        
        if attack_type == 'no_attack':
            # 无攻击，返回原始帧
            attacked = frame
            
        elif attack_type == 'jpeg_compression':
            # 模拟JPEG压缩
            # 以质量70保存为JPEG
            img = Image.fromarray((frame * 255).astype(np.uint8))
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=70)
            buffer.seek(0)
            compressed_img = Image.open(buffer)
            attacked = np.array(compressed_img).astype(np.float32) / 255.0
            
        elif attack_type == 'gaussian_noise':
            # 添加高斯噪声
            noise = np.random.normal(0, 0.05, frame.shape)
            attacked = np.clip(frame + noise, 0, 1)
            
        elif attack_type == 'rotation':
            # 旋转小角度
            angle = np.random.uniform(-10, 10)
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            attacked = cv2.warpAffine(frame, rotation_matrix, (w, h))
            
        elif attack_type == 'scaling':
            # 缩小和放大
            h, w = frame.shape[:2]
            scale = np.random.uniform(0.7, 0.9)
            new_h, new_w = int(h * scale), int(w * scale)
            frame_scaled = cv2.resize(frame, (new_w, new_h))
            attacked = cv2.resize(frame_scaled, (w, h))
            
        elif attack_type == 'cropping':
            # 裁剪中心并调整回原大小
            h, w = frame.shape[:2]
            crop_ratio = np.random.uniform(0.7, 0.9)
            crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
            start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
            cropped = frame[start_h:start_h+crop_h, start_w:start_w+crop_w]
            attacked = cv2.resize(cropped, (w, h))
        
        # 转换回张量
        attacked = torch.from_numpy(attacked.transpose(2, 0, 1)).float()
        attacked_frames.append(attacked)
    
    return torch.stack(attacked_frames).to(device)