import numpy as np
import cv2

def sift_correction(original_frame, attacked_frame):
    """
    使用SIFT特征匹配校正几何失真
    
    参数:
        original_frame: 参考帧或模板
        attacked_frame: 几何失真帧
        
    返回:
        校正后的帧
    """
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()
    
    # 寻找关键点和描述符
    kp1, des1 = sift.detectAndCompute(original_frame, None)
    kp2, des2 = sift.detectAndCompute(attacked_frame, None)
    
    # 如果找不到足够的特征，尝试使用ORB作为备选
    if len(kp1) < 10 or len(kp2) < 10:
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(original_frame, None)
        kp2, des2 = orb.detectAndCompute(attacked_frame, None)
    
    # 检查是否有足够的特征
    if len(kp1) < 4 or len(kp2) < 4:
        print("未找到足够的特征进行校正")
        return attacked_frame
    
    # 匹配特征
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # 应用比率测试
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # 如果没有足够的好匹配，返回原始帧
    if len(good_matches) < 4:
        print("未找到足够的好匹配")
        return attacked_frame
    
    # 提取匹配关键点的位置
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 使用RANSAC寻找单应性
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    # 如果找到单应性，对被攻击的帧进行变换
    if H is not None:
        h, w = original_frame.shape[:2]
        corrected_frame = cv2.warpPerspective(attacked_frame, H, (w, h))
        return corrected_frame
    else:
        print("未找到单应性")
        return attacked_frame