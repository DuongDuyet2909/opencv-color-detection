import numpy as np
import cv2


def get_limits(color):
    c = np.uint8([[color]])  # Giá trị màu BGR, biểu diễn bằng mảng uint8.
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Lấy hue (sắc độ) của màu mục tiêu trong HSV.

    # Xử lý hue gần điểm nối vòng của dải màu đỏ; mỗi nhánh chỉ trả một khoảng.
    if hue >= 165:  # Nhánh gần đầu trên của dải hue màu đỏ.
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # Nhánh gần đầu dưới của dải hue màu đỏ.
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit
