import argparse
import os
import torch
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from models.watermark import TextureDrivenWatermark
from utils.dataset import VideoWatermarkDataset
from train import train_model
from test import test_model
from utils.visualization import visualize_results

def main():
    parser = argparse.ArgumentParser(description='面向短视频平台的纹理驱动视频水印方法')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'ablation'], 
                       default='train', help='运行模式')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--watermark_length', type=int, default=32, help='水印长度')
    parser.add_argument('--frame_count', type=int, default=10, help='每个视频提取的帧数')
    parser.add_argument('--train_videos', type=str, default='data/train', help='训练视频目录')
    parser.add_argument('--val_videos', type=str, default='data/val', help='验证视频目录')
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
    
    # 图像转换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    if args.mode == 'train':
        # 确保数据目录存在
        for dir_path in [args.train_videos, args.val_videos]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"创建目录: {dir_path}")
        
        # 获取视频路径
        train_videos = [os.path.join(args.train_videos, f) for f in os.listdir(args.train_videos) 
                        if f.endswith(('.mp4', '.avi'))]
        val_videos = [os.path.join(args.val_videos, f) for f in os.listdir(args.val_videos) 
                      if f.endswith(('.mp4', '.avi'))]
        
        if len(train_videos) == 0 or len(val_videos) == 0:
            raise ValueError("未找到视频文件，请检查路径是否正确")
        
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
        
        # 初始化优化器
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
        torch.save(model.state_dict(), args.model_path)
        print(f"最终模型已保存至: {args.model_path}")
    
    elif args.mode == 'test' or args.mode == 'ablation':
        # 确保测试目录存在
        if not os.path.exists(args.test_videos):
            os.makedirs(args.test_videos, exist_ok=True)
            print(f"创建目录: {args.test_videos}")
        
        # 获取视频路径
        test_videos = [os.path.join(args.test_videos, f) for f in os.listdir(args.test_videos) 
                       if f.endswith(('.mp4', '.avi'))]
        
        if len(test_videos) == 0:
            raise ValueError("未找到视频文件，请检查路径是否正确")
        
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
        
        if args.mode == 'test':
            # 测试模型
            print("开始测试...")
            attack_types = ['no_attack', 'jpeg_compression', 'gaussian_noise', 
                          'rotation', 'scaling', 'cropping']
            test_results = test_model(model, test_loader, attack_types, device=device)
            
            # 可视化结果
            visualize_results(model, test_loader, test_results)
            print("测试完成，结果已保存")
        
        else:  # ablation mode
            # 导入消融研究函数
            from ablation import ablation_study
            
            # 执行消融研究
            print("开始消融研究...")
            ablation_results = ablation_study(model, test_loader, device=device)
            print("消融研究完成，结果已保存")

if __name__ == "__main__":
    main()