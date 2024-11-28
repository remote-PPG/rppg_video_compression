# 判断是否无损压缩

import cv2
import numpy as np

# 读取视频
video1_path = 'D:\\dataset\\compression\\ZJXU\\stationary\\20240524101350\\usb_camera-20240524101350.avi'
video2_path = 'D:\\dataset\\compression\\ZJXU\\stationary\\20240524101350\\libx265.mp4'

cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error opening video streams")
    exit()

max_diff = -1

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        break
    
    # 检查两个视频帧的尺寸是否一致
    if frame1.shape != frame2.shape:
        print("Frame dimensions do not match")
        break
    
    # 计算帧差
    diff = cv2.absdiff(frame1, frame2)
    
    # 找到差值中的最大值
    current_max_diff = np.max(diff)
    if current_max_diff > max_diff:
        max_diff = current_max_diff

# 释放视频捕获对象
cap1.release()
cap2.release()

print(f"最大差值: {max_diff}")
