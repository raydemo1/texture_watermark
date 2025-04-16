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
import torchvision


from models.watermark import TextureDrivenWatermark
from utils.dataset import VideoWatermarkDataset

# def train_model(model, train_loader, val_loader, optimizer, num_epochs=50, device='cuda'):
#     """
#     训练水印模型

#     参数:
#         model: TextureDrivenWatermark模型
#         train_loader: 训练数据加载器
#         val_loader: 验证数据加载器
#         optimizer: 优化器
#         num_epochs: 训练轮数
#         device: 计算设备
#     """
#     model = model.to(device)

#     # 图像质量的MSE损失
#     mse_criterion = nn.MSELoss()
#     # 纹理掩码质量的BCE损失
#     bce_criterion = nn.BCELoss()

#     # 训练循环
#     for epoch in range(num_epochs):
#         model.train()
#         epoch_loss = 0.0

#         for frames, watermarks in train_loader:
#             frames = frames.to(device)

#             # 清除梯度
#             optimizer.zero_grad()

#             # 前向传递
#             watermarked_frames, texture_masks = model(frames, watermarks)

#             # 计算图像质量损失（面向PSNR的MSE）
#             quality_loss = mse_criterion(watermarked_frames, frames)

#             # 计算水印鲁棒性损失
#             robust_loss = 0.0
#             # 提取水印并计算准确度
#             for i, watermark in enumerate(watermarks):
#                 watermark_tensor = torch.tensor(watermark).float().to(device)
#                 extracted = model.extract_watermark(watermarked_frames[i:i+1], len(watermark))[0]
#                 extracted_tensor = torch.tensor(extracted).float().to(device)
#                 robust_loss += F.binary_cross_entropy(extracted_tensor, watermark_tensor)

#             robust_loss /= len(watermarks)

#             # 计算纹理掩码质量损失（可选）
#             # 简单的正则化，偏好高纹理区域
#             # 通过鼓励掩码的平均值在0.3左右来模拟这一点
#             mask_loss = torch.abs(texture_masks.mean() - 0.3)

#             # 带权重的总损失（按计划）
#             total_loss = 0.7 * quality_loss + 0.3 * robust_loss + 0.1 * mask_loss

#             # 反向传递和优化
#             total_loss.backward()
#             optimizer.step()

#             epoch_loss += total_loss.item()

#         # 验证阶段
#         model.eval()
#         val_psnr = 0.0
#         val_accuracy = 0.0

#         with torch.no_grad():
#             for frames, watermarks in val_loader:
#                 frames = frames.to(device)

#                 # 前向传递
#                 watermarked_frames, _ = model(frames, watermarks)

#                 # 计算PSNR
#                 for i in range(frames.size(0)):
#                     original = frames[i].detach().cpu().numpy().transpose(1, 2, 0)
#                     watermarked = watermarked_frames[i].detach().cpu().numpy().transpose(1, 2, 0)
#                     val_psnr += psnr(original, watermarked)

#                 # 计算水印提取准确度
#                 for i, watermark in enumerate(watermarks):
#                     extracted = model.extract_watermark(watermarked_frames[i:i+1], len(watermark))[0]
#                     accuracy = np.mean(watermark == extracted)
#                     val_accuracy += accuracy

#         val_psnr /= len(val_loader.dataset)
#         val_accuracy /= len(val_loader.dataset)

#         print(f'轮次 {epoch+1}/{num_epochs}, 损失: {epoch_loss/len(train_loader):.4f}, '
#               f'验证PSNR: {val_psnr:.2f} dB, 验证准确度: {val_accuracy:.4f}')

#         # 保存检查点
#         if (epoch + 1) % 10 == 0:
#             checkpoint_path = f"checkpoints/watermark_model_epoch_{epoch+1}.pth"
#             os.makedirs('checkpoints', exist_ok=True)
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': epoch_loss/len(train_loader),
#             }, checkpoint_path)
#             print(f"模型检查点已保存至: {checkpoint_path}")


#     return model
def train_model(
    model, train_loader, val_loader, optimizer, num_epochs=50, device="cuda"
):
    """
    训练水印模型，并添加调试信息
    """
    model = model.to(device)

    # 图像质量的MSE损失
    mse_criterion = nn.MSELoss()
    # 纹理掩码质量的BCE损失
    bce_criterion = nn.BCELoss()

    # 添加一个TensorBoard记录器（可选）
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_quality_loss = 0.0
        epoch_robust_loss = 0.0

        # 在每个epoch开始时获取一个批次用于可视化
        debug_frames, debug_watermarks = next(iter(train_loader))
        debug_frames = debug_frames.to(device)

        for batch_idx, (frames, watermarks) in enumerate(train_loader):
            frames = frames.to(device)

            # 调试信息：打印帧和水印的形状和值范围
            if batch_idx == 0:
                print(
                    f"[Epoch {epoch+1}] 帧形状: {frames.shape}, 范围: [{frames.min().item():.4f}, {frames.max().item():.4f}]"
                )
                print(
                    f"水印形状: {np.array(watermarks).shape if isinstance(watermarks, list) else watermarks.shape}"
                )
                if isinstance(watermarks, list):
                    print(f"水印示例: {watermarks[0][:10]}")
                else:
                    print(f"水印示例: {watermarks[0, :10]}")

            # 清除梯度
            optimizer.zero_grad()

            # 前向传递
            watermarked_frames, texture_masks = model(frames, watermarks)

            # 调试信息：检查水印帧是否有异常值
            if batch_idx == 0:
                print(
                    f"水印帧范围: [{watermarked_frames.min().item():.4f}, {watermarked_frames.max().item():.4f}]"
                )
                print(
                    f"纹理掩码范围: [{texture_masks.min().item():.4f}, {texture_masks.max().item():.4f}]"
                )

                # 检查是否有NaN值
                if torch.isnan(watermarked_frames).any():
                    print("警告：水印帧包含NaN值！")

                # 保存图像以便可视化（每10个epoch）
                if epoch % 10 == 0:
                    import matplotlib.pyplot as plt
                    import os

                    # 创建目录
                    os.makedirs("debug_images", exist_ok=True)

                    # 取第一帧进行可视化
                    orig_img = frames[0].detach().cpu().permute(1, 2, 0).numpy()
                    wmk_img = (
                        watermarked_frames[0].detach().cpu().permute(1, 2, 0).numpy()
                    )
                    mask_img = texture_masks[0, 0].detach().cpu().numpy()

                    # 确保值范围在0-1之间
                    orig_img = np.clip(orig_img, 0, 1)
                    wmk_img = np.clip(wmk_img, 0, 1)

                    # 计算差异图（放大10倍以便可视化）
                    diff_img = np.abs(orig_img - wmk_img) * 10

                    plt.figure(figsize=(15, 10))

                    plt.subplot(221)
                    plt.imshow(orig_img)
                    plt.title("原始帧")
                    plt.axis("off")

                    plt.subplot(222)
                    plt.imshow(wmk_img)
                    plt.title("水印帧")
                    plt.axis("off")

                    plt.subplot(223)
                    plt.imshow(mask_img, cmap="jet")
                    plt.title("纹理掩码")
                    plt.axis("off")

                    plt.subplot(224)
                    plt.imshow(diff_img)
                    plt.title("差异 (x10)")
                    plt.axis("off")

                    plt.tight_layout()
                    plt.savefig(f"debug_images/epoch_{epoch+1}_batch_{batch_idx}.png")
                    plt.close()

            # 计算图像质量损失（面向PSNR的MSE）
            quality_loss = mse_criterion(watermarked_frames, frames)

            # 调试信息：打印MSE值
            if batch_idx == 0:
                mse_value = quality_loss.item()
                psnr_value = 10 * np.log10(1.0 / mse_value) if mse_value > 0 else 100
                print(f"MSE: {mse_value:.6f}, 估计PSNR: {psnr_value:.2f} dB")

            # 计算水印鲁棒性损失
            robust_loss = 0.0
            # 提取水印并计算准确度
            watermark_accuracy = []

            for i, watermark in enumerate(watermarks):
                watermark_tensor = (
                    torch.from_numpy(np.array(watermark)).float().to(device)
                    if isinstance(watermark, list)
                    else watermark.float().to(device)
                )

                # 提取水印
                extracted = model.extract_watermark(
                    watermarked_frames[i : i + 1], len(watermark)
                )[0]
                extracted_tensor = torch.tensor(extracted).float().to(device)

                # 计算位准确度
                bit_accuracy = (
                    (extracted_tensor == watermark_tensor).float().mean().item()
                )
                watermark_accuracy.append(bit_accuracy)

                # 损失计算
                robust_loss += F.binary_cross_entropy(
                    extracted_tensor, watermark_tensor
                )

            robust_loss /= len(watermarks)
            avg_accuracy = np.mean(watermark_accuracy)

            # 调试信息：打印水印提取准确度
            if batch_idx == 0:
                print(f"水印提取准确度: {avg_accuracy:.4f}")

                # 对于第一个样本显示原始和提取的水印
                if isinstance(watermarks, list):
                    orig_wm = watermarks[0]
                else:
                    orig_wm = watermarks[0].cpu().numpy()

                ext_wm = model.extract_watermark(watermarked_frames[0:1], len(orig_wm))[
                    0
                ]
                print(f"原始水印 (前20位): {orig_wm[:20]}")
                print(f"提取的水印 (前20位): {ext_wm[:20]}")

            # 计算纹理掩码质量损失（可选）
            # 简单的正则化，偏好高纹理区域
            # 通过鼓励掩码的平均值在0.3左右来模拟这一点
            mask_loss = torch.abs(texture_masks.mean() - 0.3)

            # 带权重的总损失（按计划）
            total_loss = 0.7 * quality_loss + 0.3 * robust_loss + 0.1 * mask_loss

            # 反向传递和优化
            total_loss.backward()

            # 调试：检查和打印梯度统计
            if batch_idx == 0:
                # 获取所有参数的梯度统计
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm**0.5
                print(f"梯度范数: {total_norm:.4f}")

                # 检查梯度是否有NaN
                has_nan = False
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"警告：参数 {name} 的梯度包含NaN值！")
                        has_nan = True
                if has_nan:
                    print("发现NaN梯度，跳过此批次的优化步骤")
                    optimizer.zero_grad()
                    continue

            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_quality_loss += quality_loss.item()
            epoch_robust_loss += robust_loss.item()

        # 记录训练指标
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_quality_loss = epoch_quality_loss / len(train_loader)
        avg_robust_loss = epoch_robust_loss / len(train_loader)

        writer.add_scalar("Loss/train_total", avg_epoch_loss, epoch)
        writer.add_scalar("Loss/train_quality", avg_quality_loss, epoch)
        writer.add_scalar("Loss/train_robust", avg_robust_loss, epoch)

        # 验证阶段
        model.eval()
        val_psnr = 0.0
        val_accuracy = 0.0
        val_samples = 0

        with torch.no_grad():
            for frames, watermarks in val_loader:
                frames = frames.to(device)
                batch_size = frames.size(0)
                val_samples += batch_size

                # 前向传递
                watermarked_frames, _ = model(frames, watermarks)

                # 计算PSNR
                for i in range(frames.size(0)):
                    original = frames[i].detach().cpu().numpy().transpose(1, 2, 0)
                    watermarked = (
                        watermarked_frames[i].detach().cpu().numpy().transpose(1, 2, 0)
                    )
                    from skimage.metrics import peak_signal_noise_ratio as psnr

                    # 检查值范围，确保在0-1之间
                    if original.max() > 1.0 or original.min() < 0.0:
                        print(
                            f"警告：原始帧的值超出范围 [{original.min()}, {original.max()}]"
                        )
                        original = np.clip(original, 0, 1)

                    if watermarked.max() > 1.0 or watermarked.min() < 0.0:
                        print(
                            f"警告：水印帧的值超出范围 [{watermarked.min()}, {watermarked.max()}]"
                        )
                        watermarked = np.clip(watermarked, 0, 1)

                    # 检查是否包含NaN
                    if np.isnan(original).any() or np.isnan(watermarked).any():
                        print("警告：帧包含NaN值，跳过PSNR计算")
                        continue

                    try:
                        # 计算PSNR
                        psnr_value = psnr(original, watermarked, data_range=1.0)
                        val_psnr += psnr_value
                    except Exception as e:
                        print(f"计算PSNR时出错: {e}")
                        print(f"原始帧范围: [{original.min()}, {original.max()}]")
                        print(f"水印帧范围: [{watermarked.min()}, {watermarked.max()}]")

                # 计算水印提取准确度
                for i, watermark in enumerate(watermarks):
                    extracted = model.extract_watermark(
                        watermarked_frames[i : i + 1], len(watermark)
                    )[0]
                    accuracy = np.mean(np.array(watermark) == extracted)
                    val_accuracy += accuracy

        val_psnr /= val_samples
        val_accuracy /= val_samples

        writer.add_scalar("Metrics/val_psnr", val_psnr, epoch)
        writer.add_scalar("Metrics/val_accuracy", val_accuracy, epoch)

        print(
            f"轮次 {epoch+1}/{num_epochs}, 损失: {avg_epoch_loss:.4f}, "
            f"验证PSNR: {val_psnr:.2f} dB, 验证准确度: {val_accuracy:.4f}"
        )

        # 在TensorBoard中添加图像（可选）
        if epoch % 10 == 0:
            # 可视化debug样本
            with torch.no_grad():
                debug_watermarked, debug_masks = model(
                    debug_frames[:4],
                    (
                        [watermarks[0] for _ in range(4)]
                        if isinstance(watermarks, list)
                        else watermarks[:4]
                    ),
                )

                # 原始帧
                img_grid = torchvision.utils.make_grid(debug_frames[:4], normalize=True)
                writer.add_image("Images/original", img_grid, epoch)

                # 水印帧
                img_grid = torchvision.utils.make_grid(
                    debug_watermarked[:4], normalize=True
                )
                writer.add_image("Images/watermarked", img_grid, epoch)

                # 纹理掩码
                mask_grid = torchvision.utils.make_grid(debug_masks[:4], normalize=True)
                writer.add_image("Images/texture_mask", mask_grid, epoch)

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoints/watermark_model_epoch_{epoch+1}.pth"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                    "psnr": val_psnr,
                    "accuracy": val_accuracy,
                },
                checkpoint_path,
            )
            print(f"模型检查点已保存至: {checkpoint_path}")

    # 关闭TensorBoard写入器
    writer.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="训练纹理驱动视频水印模型")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--watermark_length", type=int, default=32, help="水印长度")
    parser.add_argument(
        "--frame_count", type=int, default=10, help="每个视频提取的帧数"
    )
    parser.add_argument(
        "--train_videos", type=str, default="data/train", help="训练视频目录"
    )
    parser.add_argument(
        "--val_videos", type=str, default="data/val", help="验证视频目录"
    )
    args = parser.parse_args()

    # 设置随机种子以增加可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 获取视频路径
    train_videos = [
        os.path.join(args.train_videos, f)
        for f in os.listdir(args.train_videos)
        if f.endswith((".mp4", ".avi"))
    ]
    val_videos = [
        os.path.join(args.val_videos, f)
        for f in os.listdir(args.val_videos)
        if f.endswith((".mp4", ".avi"))
    ]

    if len(train_videos) == 0 or len(val_videos) == 0:
        raise ValueError("未找到视频文件，请检查路径是否正确")

    # 图像转换
    transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    # 创建数据集
    train_dataset = VideoWatermarkDataset(
        train_videos,
        frame_count=args.frame_count,
        transform=transform,
        watermark_length=args.watermark_length,
    )
    val_dataset = VideoWatermarkDataset(
        val_videos,
        frame_count=args.frame_count,
        transform=transform,
        watermark_length=args.watermark_length,
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # 初始化模型
    model = TextureDrivenWatermark()

    # 初始化优化器（按计划使用带动量的SGD）
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
    )

    # 训练模型
    print("开始训练...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        num_epochs=args.epochs,
        device=device,
    )

    # 保存训练好的模型
    os.makedirs("models_saved", exist_ok=True)
    model_path = "models_saved/texture_driven_watermark_final.pth"
    torch.save(model.state_dict(), model_path)
    print(f"最终模型已保存至: {model_path}")


if __name__ == "__main__":
    main()
