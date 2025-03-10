import torch
import numpy as np
import os
import argparse
import random
from torch.utils.data import DataLoader
from torchvision import transforms

from models.watermark import TextureDrivenWatermark
from utils.dataset import VideoWatermarkDataset
from utils.attacks import apply_attack
from utils.visualization import plot_comparison

def apply_traditional_dct(frames, watermarks):
    """应用传统DCT水印嵌入（无注意力机制）"""
    device = frames.device
    watermarked_frames = []
    
    for i in range(frames.size(0)):
        frame = frames[i].detach().cpu().numpy().transpose(1, 2, 0)
        # 转换为YUV
        y = 0.299 * frame[:,:,0] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,2]
        
        # 创建固定掩码（无纹理适应性）
        fixed_mask = np.ones_like(y) * 0.5
        
        # 嵌入水印
        watermark = watermarks[i]
        from models.dct import dct_embed
        marked_y = dct_embed(y, watermark, fixed_mask)
        
        # 创建嵌入水印的帧
        marked_frame = frame.copy()
        marked_frame[:,:,0] = marked_y * 0.299
        marked_frame[:,:,1] = marked_y * 0.587
        marked_frame[:,:,2] = marked_y * 0.114
        
        # 转换回张量
        marked_tensor = torch.from_numpy(marked_frame.transpose(2, 0, 1)).float()
        watermarked_frames.append(marked_tensor)
    
    return torch.stack(watermarked_frames).to(device)

def extract_traditional_dct(frames, watermark_length):
    """从传统DCT嵌入中提取水印"""
    y_channels = []
    
    for i in range(frames.size(0)):
        frame = frames[i].detach().cpu().numpy().transpose(1, 2, 0)
        # 提取Y通道
        y = 0.299 * frame[:,:,0] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,2]
        y_channels.append(y)
    
    # 提取水印
    extracted_watermarks = []
    from models.dct import dct_extract
    for y_channel in y_channels:
        extracted = dct_extract(y_channel, watermark_length)
        extracted_watermarks.append(extracted)
    
    return extracted_watermarks

def ablation_study(model, test_loader, device='cuda'):
    """执行消融研究，比较不同模块的贡献"""
    model = model.to(device)
    
    # 完整模型（双注意力）
    full_model = model
    
    # 仅空间注意力（禁用通道注意力）
    spatial_only = TextureDrivenWatermark()
    spatial_only.load_state_dict(model.state_dict())
    # 将所有通道权重设为相等
    spatial_only.channel_attention.y_weight.data.fill_(0.33)
    spatial_only.channel_attention.u_weight.data.fill_(0.33)
    spatial_only.channel_attention.v_weight.data.fill_(0.33)
    spatial_only = spatial_only.to(device)
    
    # 传统DCT（无注意力）
    traditional_dct = lambda x, w: (apply_traditional_dct(x, w), None)
    
    models = {
        "完整模型（双注意力）": full_model,
        "仅空间注意力": spatial_only,
        "传统DCT（无注意力）": traditional_dct
    }
    
    results = {}
    
    # 测试每个模型变体
    for name, model_variant in models.items():
        print(f"\n测试 {name}")
        attack_types = ['no_attack', 'jpeg_compression', 'rotation']
        
        model_results = {attack: {'psnr': [], 'accuracy': []} for attack in attack_types}
        
        with torch.no_grad():
            for frames, watermarks in test_loader:
                frames = frames.to(device)
                
                # 嵌入水印
                if name == "传统DCT（无注意力）":
                    watermarked_frames, _ = model_variant(frames, watermarks)
                else:
                    watermarked_frames, _ = model_variant(frames, watermarks)
                
                # 用不同攻击测试
                for attack in attack_types:
                    attacked_frames = apply_attack(watermarked_frames, attack)
                    
                    # 计算PSNR
                    for i in range(frames.size(0)):
                        original = frames[i].detach().cpu().numpy().transpose(1, 2, 0)
                        attacked = attacked_frames[i].detach().cpu().numpy().transpose(1, 2, 0)
                        from skimage.metrics import peak_signal_noise_ratio as psnr
                        model_results[attack]['psnr'].append(psnr(original, attacked))
                    
                    # 提取水印并计算准确度
                    for i, watermark in enumerate(watermarks):
                        if name == "传统DCT（无注意力）":
                            # 传统DCT提取
                            extracted = extract_traditional_dct(attacked_frames[i:i+1], len(watermark))[0]
                        else:
                            # 使用我们的模型提取
                            needs_correction = attack in ['rotation', 'scaling', 'cropping']
                            reference = frames[i:i+1] if needs_correction else None
                            extracted = model_variant.extract_watermark(
                                attacked_frames[i:i+1], 
                                len(watermark),
                                apply_correction=needs_correction, 
                                reference_frame=reference
                            )[0]
                        
                        accuracy = np.mean(watermark == extracted)
                        model_results[attack]['accuracy'].append(accuracy)
        
        # 计算平均结果
        for attack in attack_types:
            avg_psnr = np.mean(model_results[attack]['psnr'])
            avg_accuracy = np.mean(model_results[attack]['accuracy'])
            print(f'攻击: {attack}, PSNR: {avg_psnr:.2f} dB, 准确度: {avg_accuracy:.4f}')
        
        results[name] = model_results
    
    # 可视化结果
    plot_comparison(results)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='进行纹理驱动视频水印的消融研究')
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
    
    # 执行消融研究
    print("开始消融研究...")
    ablation_results = ablation_study(model, test_loader, device=device)
    print("消融研究完成，结果已保存")

if __name__ == "__main__":
    main()