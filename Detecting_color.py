import cv2
from PIL import Image

from util import get_limits


yellow = [0, 255, 255]  # yellow in BGR colorspace
cap = cv2.VideoCapture(0) # mở camera
while True:
    ret, frame = cap.read() # đọc khung hình từ camera

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # chuyển đổi khung hình từ BGR sang HSV

    lowerLimit, upperLimit = get_limits(color=yellow) # lấy giới hạn màu từ hàm get_limits

    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit) # tạo mặt nạ nhị phân từ khung hình HSV và giới hạn màu

    mask_ = Image.fromarray(mask) # chuyển đổi mặt nạ nhị phân sang định dạng ảnh PIL

    bbox = mask_.getbbox() # lấy bounding box của vùng màu vàng trong mặt nạ nhị phân

    if bbox is not None: # nếu bounding box tồn tại, vẽ hình chữ nhật xung quanh vùng màu vàng trong khung hình gốc
        x1, y1, x2, y2 = bbox

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow('frame', frame) # hiển thị khung hình với bounding box

    if cv2.waitKey(1) & 0xFF == ord('q'): # nếu nhấn phím 'q', thoát khỏi vòng lặp while
        break

cap.release() # giải phóng camera

cv2.destroyAllWindows() # đóng tất cả các cửa sổ OpenCV