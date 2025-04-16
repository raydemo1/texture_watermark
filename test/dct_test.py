import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path


# 简化版DCT嵌入函数（使用固定步长）
def simple_dct_embed(frame, watermark, block_size=8, step=10):
    """简化版DCT嵌入（无注意力机制，固定步长）"""
    marked_frame = frame.copy()
    height, width = frame.shape[:2]

    # 确保水印是一维整数数组
    watermark = np.asarray(watermark).flatten().astype(np.int32)

    # 预处理水印
    total_blocks = (height // block_size) * (width // block_size)
    watermark_bits = np.tile(watermark, (total_blocks // len(watermark) + 1))[
        :total_blocks
    ]
    watermark_idx = 0

    # 处理每个块
    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            # 提取块
            block = frame[y : y + block_size, x : x + block_size].astype(np.float32)

            # 应用DCT
            dct_block = cv2.dct(block)

            # 选择中频系数
            u, v = 3, 4

            # 嵌入水印位（使用固定步长）
            bit_value = int(watermark_bits[watermark_idx])

            if bit_value == 1:
                dct_block[u, v] = np.floor(dct_block[u, v] / step) * step + step / 2
            else:
                dct_block[u, v] = np.floor(dct_block[u, v] / step) * step

            # 应用逆DCT
            marked_block = cv2.idct(dct_block)

            # 更新帧
            marked_frame[y : y + block_size, x : x + block_size] = marked_block

            watermark_idx += 1

    return marked_frame


# 简化版DCT提取函数
def simple_dct_extract(frame, watermark_length, block_size=8, step=10):
    """简化版DCT提取（固定步长）"""
    height, width = frame.shape[:2]
    all_extracted_bits = []

    # 处理每个块
    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            block = frame[y : y + block_size, x : x + block_size].astype(np.float32)
            dct_block = cv2.dct(block)

            # 使用相同系数位置
            u, v = 3, 4
            coef_value = dct_block[u, v]

            # 提取位
            mod_value = coef_value % step
            if mod_value > step / 4 and mod_value < 3 * step / 4:
                bit = 1
            else:
                bit = 0

            all_extracted_bits.append(bit)

    # 提取水印
    num_complete_watermarks = len(all_extracted_bits) // watermark_length
    if num_complete_watermarks > 0:
        reshaped_bits = np.array(
            all_extracted_bits[: num_complete_watermarks * watermark_length]
        )
        reshaped_bits = reshaped_bits.reshape(num_complete_watermarks, watermark_length)
        extracted_watermark = np.round(np.mean(reshaped_bits, axis=0)).astype(int)
    else:
        # 提取的位不足一个完整水印
        extracted_watermark = np.array(all_extracted_bits[:watermark_length])

    return extracted_watermark


# 用于测试的主函数
def test_simple_watermarking(image_folder="./images"):
    # 1. 从指定文件夹读取所有图片
    image_paths = list(Path(image_folder).glob("*.*"))
    image_paths = [
        str(p) for p in image_paths if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_folder}")

    # 创建结果保存目录
    output_dir = Path("watermark_results")
    output_dir.mkdir(exist_ok=True)

    # 2. 创建水印
    watermark_length = 32
    watermark = np.random.randint(0, 2, size=watermark_length)
    print("原始水印:", watermark)

    # 3. 测试不同步长
    steps = [5, 10, 20, 30]

    for img_path in image_paths:
        plt.figure(figsize=(15, 10))

        # 读取并预处理图像
        img_name = Path(img_path).stem
        image = Image.open(img_path).convert("L")  # 转为灰度
        test_image = np.array(image)

        # 确保尺寸是block_size的倍数
        block_size = 8
        h, w = test_image.shape
        test_image = test_image[
            : h // block_size * block_size, : w // block_size * block_size
        ]

        for i, step in enumerate(steps):
            # 嵌入水印
            watermarked_image = simple_dct_embed(test_image, watermark, step=step)

            # 计算PSNR
            mse = np.mean((test_image - watermarked_image) ** 2)
            psnr = 10 * np.log10(255**2 / mse) if mse != 0 else 100

            # 提取水印
            extracted_watermark = simple_dct_extract(
                watermarked_image, watermark_length, step=step
            )

            # 计算准确度
            accuracy = np.mean(watermark == extracted_watermark)

            # 显示结果
            plt.subplot(len(steps), 2, i * 2 + 1)
            plt.imshow(watermarked_image, cmap="gray")
            plt.title(f"Step={step}, PSNR={psnr:.2f}dB")
            plt.axis("off")

            plt.subplot(len(steps), 2, i * 2 + 2)
            diff = np.abs(test_image - watermarked_image) * 10
            plt.imshow(diff, cmap="hot")
            plt.title(f"Difference Map, Acc={accuracy:.2f}")
            plt.axis("off")

        # 保存结果
        result_path = output_dir / f"{img_name}_results.png"
        plt.tight_layout()
        plt.savefig(result_path)
        plt.close()
        print(f"Saved results for {img_name} to {result_path}")


# 执行测试
if __name__ == "__main__":
    test_simple_watermarking(image_folder="./images")
