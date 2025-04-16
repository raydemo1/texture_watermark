import numpy as np
import cv2


def classify_block(block):
    """将块分类为边缘、纹理或平滑区域"""
    # 计算梯度特征
    sobel_x = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # 计算统计特征
    mean_gradient = np.mean(gradient_magnitude)
    std_block = np.std(block)

    # 分类规则
    if mean_gradient > 20:  # 阈值需要根据图像特性调整
        return "edge"
    elif std_block > 10:
        return "texture"
    else:
        return "smooth"


def adaptive_block_embed(block, bit_value, block_type):
    """根据块类型选择不同的嵌入策略"""
    # 应用DCT变换
    dct_block = cv2.dct(block.astype(np.float32))

    if block_type == "edge":
        # 边缘区域：使用较高频系数和较小步长
        u, v = 5, 5  # 高频位置
        step = 1.0  # 小步长
        # 可能使用不同的嵌入方式，例如幅相嵌入

    elif block_type == "texture":
        # 纹理区域：使用中频系数和中等步长
        u, v = 3, 4  # 中频位置
        step = 3.0  # 中等步长

    else:  # "smooth"
        # 平滑区域：可能不嵌入或使用极低步长
        u, v = 2, 2  # 较低频位置
        step = 0.5  # 极小步长

    # 执行量化嵌入
    if bit_value == 1:
        dct_block[u, v] = np.floor(dct_block[u, v] / step) * step + step / 2
    else:
        dct_block[u, v] = np.floor(dct_block[u, v] / step) * step

    # 逆DCT恢复块
    return cv2.idct(dct_block)


def dct_embed(frame, watermark, texture_mask, block_size=8):
    """
    在DCT域中嵌入水印，使用纹理自适应量化

    参数:
        frame: 输入帧（YUV，使用Y通道）- numpy数组
        watermark: 二进制水印位 - numpy数组
        texture_mask: 纹理注意力掩码（0-1值）- numpy数组
        block_size: DCT块大小（默认：8）

    返回:
        嵌入水印的帧
    """
    # 复制帧
    marked_frame = frame.copy()
    height, width = frame.shape[:2]

    # 确保水印是一维整数数组
    watermark = np.asarray(watermark).flatten().astype(np.int32)

    # 预处理水印，如果长度与块数不匹配
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

            # 分析块类型
            block_type = classify_block(block)

            # 如果是平滑区域，可以选择性跳过
            if block_type == "smooth" and np.random.random() > 0.3:
                # 70%概率跳过平滑区域
                watermark_idx += 1
                if watermark_idx >= len(watermark_bits):
                    watermark_idx = 0
                continue

            # 嵌入水印位
            bit_value = int(watermark_bits[watermark_idx])  # 直接转换为整数

            # 自适应嵌入
            marked_block = adaptive_block_embed(block, bit_value, block_type)

            # 更新帧
            marked_frame[y : y + block_size, x : x + block_size] = marked_block

            watermark_idx += 1
            if watermark_idx >= len(watermark_bits):
                watermark_idx = 0

    return marked_frame


def dct_extract(frame, watermark_length, block_size=8, step_estimate=3):
    """
    从DCT域中提取水印

    参数:
        frame: 嵌入水印的帧（YUV，使用Y通道）
        watermark_length: 原始水印的长度
        block_size: DCT块大小（应与嵌入匹配）
        step_estimate: 估计的量化步长（基础）

    返回:
        提取的水印位
    """
    height, width = frame.shape[:2]
    total_blocks = (height // block_size) * (width // block_size)

    # 初始化数组以存储从所有块提取的位
    all_extracted_bits = []

    # 处理每个块
    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            # 提取块
            block = frame[y : y + block_size, x : x + block_size].astype(np.float32)

            # 应用DCT
            dct_block = cv2.dct(block)

            # 使用与嵌入相同的系数位置
            u, v = 3, 4
            coef_value = dct_block[u, v]

            # 估计适当的步长
            # 在实际提取中，我们可能需要尝试各种步长
            # 或为不同纹理级别使用预定义的步长
            step = step_estimate

            # 通过检查系数是否更接近步长的倍数或步长的倍数加上步长/2来提取位
            mod_value = coef_value % step
            if mod_value > step / 4 and mod_value < 3 * step / 4:
                bit = 1
            else:
                bit = 0

            all_extracted_bits.append(bit)

    # 对所有块提取的位求平均以获得最终水印
    # 首先将其重新调整为每个完整水印一行
    num_complete_watermarks = len(all_extracted_bits) // watermark_length
    reshaped_bits = np.array(
        all_extracted_bits[: num_complete_watermarks * watermark_length]
    )
    reshaped_bits = reshaped_bits.reshape(num_complete_watermarks, watermark_length)

    # 对每个位位置进行多数投票
    extracted_watermark = np.round(np.mean(reshaped_bits, axis=0)).astype(int)

    return extracted_watermark
