import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
import argparse
import random
import sys

# 添加项目根目录到系统路径
sys.path.append(".")

# 导入自定义模块
from models.watermark import TextureDrivenWatermark
from models.dct import dct_embed, dct_extract
from utils.attacks import apply_attack


def rgb_to_yuv(rgb_image):
    """将RGB图像转换为YUV"""
    # OpenCV的BGR转YUV
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    yuv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YUV)
    return yuv_image


def yuv_to_rgb(yuv_image):
    """将YUV图像转换为RGB"""
    # OpenCV的YUV转BGR，然后转RGB
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image


def compute_psnr(original, watermarked):
    """计算峰值信噪比"""
    from skimage.metrics import peak_signal_noise_ratio as psnr

    return psnr(original, watermarked)


def compute_ssim(original, watermarked):
    """计算结构相似性"""
    from skimage.metrics import structural_similarity as ssim

    # 如果是RGB图像，需要设置multichannel=True
    if original.ndim == 3:
        return ssim(original, watermarked, multichannel=True)
    return ssim(original, watermarked)


def generate_binary_logo():
    """生成一个简单的二进制水印"""
    # 创建32x32的二进制水印
    logo = np.zeros((32, 32), dtype=np.uint8)

    # 在中间绘制一个十字
    logo[12:20, 8:24] = 1  # 横线
    logo[8:24, 12:20] = 1  # 竖线

    # 将2D水印展平成1D数组
    return logo.flatten()


def demo_watermark_embedding(input_image_path, output_dir, model_path=None):
    """演示水印嵌入和提取过程"""
    os.makedirs(output_dir, exist_ok=True)

    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化模型
    model = TextureDrivenWatermark()

    # 如果有预训练模型，则加载
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"已加载模型权重: {model_path}")

    model = model.to(device)
    model.eval()

    # 生成二进制水印
    watermark = generate_binary_logo()
    watermark_length = len(watermark)
    print(f"水印长度: {watermark_length}")

    # 加载图像
    original_image = Image.open(input_image_path).convert("RGB")

    # 图像预处理
    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    frame = transform(original_image).unsqueeze(0).to(device)  # 添加批次维度

    # 嵌入水印
    with torch.no_grad():
        # 计算LBP特征
        lbp_tensor = model.compute_lbp(frame)

        # 计算纹理注意力掩码
        texture_mask = model.spatial_attention(frame, lbp_tensor)

        # 嵌入水印
        watermarked_frame, _ = model.embed_watermark(frame, [watermark], texture_mask)

    # 将张量转换为numpy数组以进行可视化
    original_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
    watermarked_np = watermarked_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
    mask_np = texture_mask.squeeze(0).squeeze(0).cpu().numpy()

    # 计算图像质量指标
    psnr_value = compute_psnr(original_np, watermarked_np)
    ssim_value = compute_ssim(original_np, watermarked_np)

    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")

    # 创建包含多种攻击的字典
    attack_types = {
        "no_attack": "无攻击",
        "jpeg_compression": "JPEG压缩",
        "gaussian_noise": "高斯噪声",
        "rotation": "旋转",
        "scaling": "缩放",
    }

    # 对每种攻击提取水印并计算准确率
    attack_results = {}

    # 保存水印嵌入前后的图像和注意力掩码
    plt.figure(figsize=(18, 12))

    # 显示原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(original_np)
    plt.title("原始图像")
    plt.axis("off")

    # 显示嵌入水印的图像
    plt.subplot(2, 3, 2)
    plt.imshow(watermarked_np)
    plt.title(f"嵌入水印后 (PSNR: {psnr_value:.2f}dB)")
    plt.axis("off")

    # 显示注意力掩码
    plt.subplot(2, 3, 3)
    plt.imshow(mask_np, cmap="jet")
    plt.title("纹理注意力掩码")
    plt.axis("off")

    # 显示差异图（放大10倍以便可视化）
    plt.subplot(2, 3, 4)
    difference = np.abs(original_np - watermarked_np) * 10
    plt.imshow(difference)
    plt.title("差异 (×10)")
    plt.axis("off")

    # 为每种攻击测试提取
    row = 5
    for attack_type, attack_name in attack_types.items():
        # 应用攻击
        attacked_frame = apply_attack(watermarked_frame, attack_type)

        # 提取水印
        needs_correction = attack_type in ["rotation", "scaling", "cropping"]
        reference = frame if needs_correction else None

        extracted_watermark = model.extract_watermark(
            attacked_frame,
            watermark_length,
            apply_correction=needs_correction,
            reference_frame=reference,
        )[0]

        # 计算准确率
        accuracy = np.mean(watermark == extracted_watermark)
        attack_results[attack_type] = accuracy

        if row <= 6:  # 只显示前两种攻击的可视化
            # 显示受攻击的图像
            plt.subplot(2, 3, row)
            attacked_np = attacked_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            plt.imshow(attacked_np)
            plt.title(f"{attack_name} (准确率: {accuracy:.2f})")
            plt.axis("off")
            row += 1

            # 保存受攻击的图像
            attacked_image = Image.fromarray((attacked_np * 255).astype(np.uint8))
            attacked_image.save(os.path.join(output_dir, f"attacked_{attack_type}.png"))

    # 美化并保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "watermark_visualization.png"), dpi=300)

    # 单独保存原始和嵌入水印的图像
    original_image_output = Image.fromarray((original_np * 255).astype(np.uint8))
    watermarked_image_output = Image.fromarray((watermarked_np * 255).astype(np.uint8))
    mask_image_output = Image.fromarray((mask_np * 255).astype(np.uint8))

    original_image_output.save(os.path.join(output_dir, "original.png"))
    watermarked_image_output.save(os.path.join(output_dir, "watermarked.png"))
    mask_image_output.save(os.path.join(output_dir, "texture_mask.png"))

    # 创建水印提取结果表格
    plt.figure(figsize=(10, 6))
    attacks = list(attack_results.keys())
    accuracies = [attack_results[attack] for attack in attacks]

    plt.bar(range(len(attacks)), accuracies)
    plt.xticks(
        range(len(attacks)), [attack_types[attack] for attack in attacks], rotation=45
    )
    plt.xlabel("攻击类型")
    plt.ylabel("水印提取准确率")
    plt.title("不同攻击下的水印提取准确率")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "extraction_accuracy.png"))

    print(f"所有结果已保存到 {output_dir} 目录")
    print("\n提取准确率结果:")
    for attack_type, accuracy in attack_results.items():
        print(f"{attack_types[attack_type]}: {accuracy:.4f}")


def process_video_frames(video_path, output_dir, model_path=None, frame_count=5):
    """从视频中提取帧并应用水印"""
    os.makedirs(output_dir, exist_ok=True)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextureDrivenWatermark()

    # 如果有预训练模型，则加载
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    model.eval()

    # 生成二进制水印
    watermark = generate_binary_logo()

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    # 获取视频信息
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, frame_count_total // frame_count)

    # 图像预处理
    transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    frame_idx = 0
    frame_processed = 0

    # 创建总结图
    plt.figure(figsize=(15, 10))

    while cap.isOpened() and frame_processed < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # 转换BGR到RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 预处理
            frame_tensor = transform(frame_rgb).unsqueeze(0).to(device)

            # 嵌入水印
            with torch.no_grad():
                lbp_tensor = model.compute_lbp(frame_tensor)
                texture_mask = model.spatial_attention(frame_tensor, lbp_tensor)
                watermarked_frame, _ = model.embed_watermark(
                    frame_tensor, [watermark], texture_mask
                )

            # 转换回numpy
            original_np = frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            watermarked_np = watermarked_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            mask_np = texture_mask.squeeze(0).squeeze(0).cpu().numpy()

            # 计算PSNR
            psnr_value = compute_psnr(original_np, watermarked_np)

            # 保存图像
            original_image = Image.fromarray((original_np * 255).astype(np.uint8))
            watermarked_image = Image.fromarray((watermarked_np * 255).astype(np.uint8))
            mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))

            original_image.save(
                os.path.join(output_dir, f"frame_{frame_processed}_original.png")
            )
            watermarked_image.save(
                os.path.join(output_dir, f"frame_{frame_processed}_watermarked.png")
            )
            mask_image.save(
                os.path.join(output_dir, f"frame_{frame_processed}_mask.png")
            )

            # 在总结图中添加
            plt.subplot(frame_count, 3, frame_processed * 3 + 1)
            plt.imshow(original_np)
            plt.title(f"原始 #{frame_processed}")
            plt.axis("off")

            plt.subplot(frame_count, 3, frame_processed * 3 + 2)
            plt.imshow(watermarked_np)
            plt.title(f"水印 #{frame_processed} ({psnr_value:.1f}dB)")
            plt.axis("off")

            plt.subplot(frame_count, 3, frame_processed * 3 + 3)
            plt.imshow(mask_np, cmap="jet")
            plt.title(f"掩码 #{frame_processed}")
            plt.axis("off")

            frame_processed += 1

        frame_idx += 1

    cap.release()

    # 保存总结图
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "video_frames_summary.png"), dpi=300)
    print(f"视频帧处理完成，结果已保存到 {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="演示纹理驱动视频水印")
    parser.add_argument(
        "--mode",
        choices=["image", "video"],
        default="image",
        help="处理模式: 图像或视频",
    )
    parser.add_argument("--input", type=str, required=True, help="输入图像或视频路径")
    parser.add_argument(
        "--output_dir", type=str, default="demo_results", help="输出目录"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="预训练模型路径(可选)"
    )
    parser.add_argument("--frames", type=int, default=5, help="从视频中提取的帧数")

    args = parser.parse_args()

    if args.mode == "image":
        demo_watermark_embedding(args.input, args.output_dir, args.model_path)
    else:
        process_video_frames(args.input, args.output_dir, args.model_path, args.frames)
