import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class VideoWatermarkDataset(Dataset):
    def __init__(self, video_paths, frame_count=10, transform=None, watermark_length=32):
        self.video_paths = video_paths
        self.frame_count = frame_count
        self.transform = transform
        self.frames = []
        self.watermark_length = watermark_length
        
        # 从视频中提取帧
        self._extract_frames()
        
    def _extract_frames(self):
        for video_path in self.video_paths:
            cap = cv2.VideoCapture(video_path)
            frame_interval = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.frame_count)
            
            count = 0
            frame_idx = 0
            
            while cap.isOpened() and count < self.frame_count:
                ret, frame = cap.read()
                
                if not ret:
                    break
                    
                if frame_idx % frame_interval == 0:
                    # 将BGR转换为RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.frames.append(frame)
                    count += 1
                
                frame_idx += 1
            
            cap.release()
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        # 如果有可用的转换，则应用它们
        if self.transform:
            frame = self.transform(frame)
        
        # 生成用于训练的随机水印
        watermark = np.random.randint(0, 2, size=self.watermark_length)
        
        return frame, watermark