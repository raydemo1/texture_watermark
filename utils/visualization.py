import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_results(model, test_loader, test_results=None):
    """
    可视化水印结果和注意力图
    """
    device = next(model.parameters()).device
    model.eval()
    
    # 获取一批测试图像
    frames, watermarks = next(iter(test_loader))
    frames = frames.to(device)
    
    with torch.no_grad():
        # 获取嵌入水印的帧和注意力图
        watermarked_frames, texture_masks = model(frames, watermarks)
        
        # 转换为numpy以进行可视化
        orig_frames = frames.detach().cpu().numpy().transpose(0, 2, 3, 1)
        wmk_frames = watermarked_frames.detach().cpu().numpy().transpose(0, 2, 3, 1)
        masks = texture_masks.squeeze(1).detach().cpu().numpy()
        
        # 创建图形
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        # 显示原始帧、嵌入水印的帧、注意力掩码和差异
        for i in range(4):
            if i < len(orig_frames):
                # 原始帧
                axes[0, i].imshow(orig_frames[i])
                axes[0, i].set_title(f"原始 {i+1}")
                axes[0, i].axis('off')
                
                # 嵌入水印的帧
                axes[1, i].imshow(wmk_frames[i])
                axes[1, i].set_title(f"嵌入水印 {i+1}")
                axes[1, i].axis('off')
                
                # 注意力掩码
                axes[2, i].imshow(masks[i], cmap='jet')
                axes[2, i].set_title(f"纹理掩码 {i+1}")
                axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig("watermarking_results.png")
        plt.close()
        
        # 如果有测试结果，绘制准确度与攻击类型的关系
        if test_results:
            attack_types = list(test_results.keys())
            accuracies = [np.mean(test_results[attack]['accuracy']) for attack in attack_types]
            
            plt.figure(figsize=(10, 6))
            plt.bar(attack_types, accuracies)
            plt.ylim(0, 1.0)
            plt.title("不同攻击下的水印提取准确度")
            plt.ylabel("准确度")
            plt.xlabel("攻击类型")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("accuracy_vs_attacks.png")
            plt.close()

def plot_comparison(results):
    """绘制不同方法的比较图"""
    model_names = list(results.keys())
    attack_types = list(results[model_names[0]].keys())
    
    # 为每种攻击创建一个子图
    fig, axes = plt.subplots(1, len(attack_types), figsize=(15, 5))
    
    for i, attack in enumerate(attack_types):
        accuracies = [np.mean(results[model][attack]['accuracy']) for model in model_names]
        
        axes[i].bar(model_names, accuracies)
        axes[i].set_title(f'攻击: {attack}')
        axes[i].set_ylim(0, 1.0)
        axes[i].set_ylabel('准确度')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("method_comparison.png")
    plt.close()