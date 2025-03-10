import numpy as np
import cv2

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
    
    # 预处理水印，如果长度与块数不匹配
    total_blocks = (height // block_size) * (width // block_size)
    watermark_bits = np.tile(watermark, (total_blocks // len(watermark) + 1))[:total_blocks]
    watermark_idx = 0
    
    # 处理每个块
    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            # 提取块
            block = frame[y:y+block_size, x:x+block_size].astype(np.float32)
            
            # 计算该块的平均纹理值
            texture_value = np.mean(texture_mask[y:y+block_size, x:x+block_size])
            
            # 应用DCT
            dct_block = cv2.dct(block)
            
            # 选择用于嵌入的中频系数
            # 使用zigzag模式，选择系数(3,4)、(4,3)、(5,2)等
            # 为简单起见，这里使用(3,4)
            u, v = 3, 4
            
            # 基于纹理的自适应量化步长
            # 更高的纹理允许更大的步长（不可见性更好）
            base_step = 20
            adaptive_step = base_step + 15 * texture_value  # 随纹理增加
            
            # 嵌入水印位
            if watermark_bits[watermark_idx] == 1:
                # 使系数成为步长的倍数加上步长/2
                dct_block[u, v] = np.floor(dct_block[u, v] / adaptive_step) * adaptive_step + adaptive_step / 2
            else:
                # 使系数成为步长的倍数
                dct_block[u, v] = np.floor(dct_block[u, v] / adaptive_step) * adaptive_step
            
            # 应用逆DCT
            marked_block = cv2.idct(dct_block)
            
            # 用嵌入水印的块更新帧
            marked_frame[y:y+block_size, x:x+block_size] = marked_block
            
            watermark_idx += 1
            if watermark_idx >= len(watermark_bits):
                watermark_idx = 0
    
    return marked_frame

def dct_extract(frame, watermark_length, block_size=8, step_estimate=20):
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
            block = frame[y:y+block_size, x:x+block_size].astype(np.float32)
            
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
    reshaped_bits = np.array(all_extracted_bits[:num_complete_watermarks * watermark_length])
    reshaped_bits = reshaped_bits.reshape(num_complete_watermarks, watermark_length)
    
    # 对每个位位置进行多数投票
    extracted_watermark = np.round(np.mean(reshaped_bits, axis=0)).astype(int)
    
    return extracted_watermark