import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import random
import argparse

from models.watermark import TextureDrivenWatermark
from utils.dataset import VideoWatermarkDataset

def train_model(model, train_loader, val_loader, optimizer, num_epochs=50, device='cuda'):
    """
    训练水印模型
    
    参数:
        model: TextureDrivenWatermark模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 计算设备
    """
    model = model.to(device)
    
    # 图像质量的MSE损失
    mse_criterion = nn.MSELoss()
    # 纹理掩码质量的BCE损失
    bce_criterion = nn.BCELoss()
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for frames, watermarks in train_loader:
            frames = frames.to(device)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传递
            watermarked_frames, texture_masks = model(frames, watermarks)
            
            # 计算图像质量损失（面向PSNR的MSE）
            quality_loss = mse_criterion(watermarked_frames, frames)
            
            # 计算水印鲁棒性损失
            robust_loss = 0.0
            # 提取水印并计算准确度
            for i, watermark in enumerate(watermarks):
                watermark_tensor = torch.tensor(watermark).float().to(device)
                extracted = model.extract_watermark(watermarked_frames[i:i+1], len(watermark))[0]
                extracted_tensor = torch.tensor(extracted).float().to(device)
                robust_loss += F.binary_cross_entropy(extracted_tensor, watermark_tensor)
            
            robust_loss /= len(watermarks)
            
            # 计算纹理掩码质量损失（可选）
            # 简单的正则化，偏好高纹理区域
            # 通过鼓励掩码的平均值在0.3左右来模拟这一点
            mask_loss = torch.abs(texture_masks.mean() - 0.3)
            
            # 带权重的总损失（按计划）
            total_loss = 0.7 * quality_loss + 0.3 * robust_loss + 0.1 * mask_loss
            
            # 反向传递和优化
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        # 验证阶段
        model.eval()
        val_psnr = 0.0
        val_accuracy = 0.0
        
        with torch.no_grad():
            for frames, watermarks in val_loader:
                frames = frames.to(device)
                
                # 前向传递
                watermarked_frames, _ = model(frames, watermarks)
                
                # 计算PSNR
                for i in range(frames.size(0)):
                    original = frames[i].detach().cpu().numpy().transpose(1, 2, 0)
                    watermarked = watermarked_frames[i].detach().cpu().numpy().transpose(1, 2, 0)
                    val_psnr += psnr(original, watermarked)
                
                # 计算水印提取准确度
                for i, watermark in enumerate(watermarks):
                    extracted = model.extract_watermark(watermarked_frames[i:i+1], len(watermark))[0]
                    accuracy = np.mean(watermark == extracted)
                    val_accuracy += accuracy
        
        val_psnr /= len(val_loader.dataset)
        val_accuracy /= len(val_loader.dataset)
        
        print(f'轮次 {epoch+1}/{num_epochs}, 损失: {epoch_loss/len(train_loader):.4f}, '
              f'验证PSNR: {val_psnr:.2f} dB, 验证准确度: {val_accuracy:.4f}')
    
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoints/watermark_model_epoch_{epoch+1}.pth"
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss/len(train_loader),
            }, checkpoint_path)
            print(f"模型检查点已保存至: {checkpoint_path}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='训练纹理驱动视频水印模型')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--watermark_length', type=int, default=32, help='水印长度')
    parser.add_argument('--frame_count', type=int, default=10, help='每个视频提取的帧数')
    parser.add_argument('--train_videos', type=str, default='data/train', help='训练视频目录')
    parser.add_argument('--val_videos', type=str, default='data/val', help='验证视频目录')
    args = parser.parse_args()
    
    # 设置随机种子以增加可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 获取视频路径
    train_videos = [os.path.join(args.train_videos, f) for f in os.listdir(args.train_videos) if f.endswith(('.mp4', '.avi'))]
    val_videos = [os.path.join(args.val_videos, f) for f in os.listdir(args.val_videos) if f.endswith(('.mp4', '.avi'))]
    
    if len(train_videos) == 0 or len(val_videos) == 0:
        raise ValueError("未找到视频文件，请检查路径是否正确")
    
    # 图像转换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # 创建数据集
    train_dataset = VideoWatermarkDataset(train_videos, frame_count=args.frame_count, 
                                         transform=transform, watermark_length=args.watermark_length)
    val_dataset = VideoWatermarkDataset(val_videos, frame_count=args.frame_count, 
                                       transform=transform, watermark_length=args.watermark_length)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # 初始化模型
    model = TextureDrivenWatermark()
    
    # 初始化优化器（按计划使用带动量的SGD）
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # 训练模型
    print("开始训练...")
    model = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        num_epochs=args.epochs,
        device=device
    )
    
    # 保存训练好的模型
    os.makedirs('models_saved', exist_ok=True)
    model_path = "models_saved/texture_driven_watermark_final.pth"
    torch.save(model.state_dict(), model_path)
    print(f"最终模型已保存至: {model_path}")

if __name__ == "__main__":
    main()