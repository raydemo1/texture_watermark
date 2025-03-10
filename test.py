import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import argparse
import random

from models.watermark import TextureDrivenWatermark
from utils.dataset import VideoWatermarkDataset
from utils.attacks import apply_attack
from utils.visualization import visualize_results

def test_model(model, test_loader, attack_types=None, device='cuda'):
    """
    使用各种攻击测试水印模型
    
    参数:
        model: 训练好的TextureDrivenWatermark模型
        test_loader: 测试数据加载器
        attack_types: 要应用的攻击类型列表
        device: 计算设备
    """
    model = model.to(device)
    model.eval()
    
    if attack_types is None:
        attack_types = ['no_attack', 'jpeg_compression', 'gaussian_noise', 'rotation', 'scaling']
    
    results = {attack: {'psnr': [], 'accuracy': []} for attack in attack_types}
    
    with torch.no_grad():
        for frames, watermarks in test_loader:
            frames = frames.to(device)
            
            # 嵌入水印
            watermarked_frames, _ = model(frames, watermarks)
            
            # 用不同攻击测试
            for attack in attack_types:
                attacked_frames = apply_attack(watermarked_frames, attack)
                
                # 计算PSNR
                for i in range(frames.size(0)):
                    original = frames[i].detach().cpu().numpy().transpose(1, 2, 0)
                    attacked = attacked_frames[i].detach().cpu().numpy().transpose(1, 2, 0)
                    from skimage.metrics import peak_signal_noise_ratio as psnr
                    results[attack]['psnr'].append(psnr(original, attacked))
                
                # 提取水印并计算准确度
                for i, watermark in enumerate(watermarks):
                    # 如果需要则应用几何校正
                    needs_correction = attack in ['rotation', 'scaling', 'cropping']
                    reference = frames[i:i+1] if needs_correction else None
                    
                    extracted = model.extract_watermark(
                        attacked_frames[i:i+1], 
                        len(watermark),
                        apply_correction=needs_correction, 
                        reference_frame=reference
                    )[0]
                    
                    accuracy = np.mean(watermark == extracted)
                    results[attack]['accuracy'].append(accuracy)
    
    # 计算平均结果
    for attack in attack_types:
        avg_psnr = np.mean(results[attack]['psnr'])
        avg_accuracy = np.mean(results[attack]['accuracy'])
        print(f'攻击: {attack}, PSNR: {avg_psnr:.2f} dB, 准确度: {avg_accuracy:.4f}')
    
    return results

def main():
    parser = argparse.ArgumentParser(description='测试纹理驱动视频水印模型')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--watermark_length', type=int, default=32, help='水印长度')
    parser.add_argument('--frame_count', type=int, default=10, help='每个视频提取的帧数')
    parser.add_argument('--test_videos', type=str, default='data/test', help='测试视频目录')
    parser.add_argument('--model_path', type=str, default='models_saved/texture_driven_watermark_final.pth', 
                        help='模型权重路径')
    args = parser.parse_args()
    
    # 设置随机种子以增加可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 获取视频路径
    test_videos = [os.path.join(args.test_videos, f) for f in os.listdir(args.test_videos) if f.endswith(('.mp4', '.avi'))]
    
    if len(test_videos) == 0:
        raise ValueError("未找到视频文件，请检查路径是否正确")
    
    # 图像转换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # 创建数据集
    test_dataset = VideoWatermarkDataset(test_videos, frame_count=args.frame_count, 
                                       transform=transform, watermark_length=args.watermark_length)
    
    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 初始化模型
    model = TextureDrivenWatermark()
    
    # 加载训练好的模型权重
    if not os.path.exists(args.model_path):
        raise ValueError(f"未找到模型权重文件: {args.model_path}")
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"已加载模型权重: {args.model_path}")
    
    # 测试模型
    print("开始测试...")
    attack_types = ['no_attack', 'jpeg_compression', 'gaussian_noise', 
                   'rotation', 'scaling', 'cropping']
    test_results = test_model(model, test_loader, attack_types, device=device)
    
    # 可视化一些结果
    visualize_results(model, test_loader, test_results)
    print("测试完成，结果已保存")

if __name__ == "__main__":
    main()