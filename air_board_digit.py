"""
Luồng chính:
1. Mở camera và dùng MediaPipe Hands để tìm các mốc bàn tay.
2. Dùng đầu ngón trỏ như một cây bút để vẽ lên canvas.
3. Khi nhấn R, tiền xử lý hình vẽ thành ảnh 28x28.
4. Đưa ảnh vào mô hình CNN đã huấn luyện để nhận dạng chữ số.

"""
import cv2 # OpenCV phụ trách camera, xử lý ảnh, vẽ đường và hiển thị cửa sổ.
import mediapipe as mp # MediaPipe cung cấp mô hình phát hiện và theo dõi 21 landmark của bàn tay.
import numpy as np # NumPy dùng để tạo canvas, thao tác mảng ảnh và tìm lớp có xác suất lớn nhất.
import tensorflow as tf # TensorFlow/Keras dùng để tải và chạy mô hình nhận dạng chữ số.
from pathlib import Path
# Tải mô hình CNN đã được train từ file HDF5.
# compile=False vì chương trình chỉ suy luận, không tiếp tục train hay đánh giá loss.


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "digit_model.h5"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Lấy module Hands của MediaPipe để tạo bộ phát hiện/theo dõi bàn tay.
mp_hands = mp.solutions.hands

# Lấy tiện ích vẽ landmark và các đường nối giữa landmark lên frame.
mp_draw = mp.solutions.drawing_utils

# Mở camera có chỉ số 0, thường là webcam mặc định của máy.
cap = cv2.VideoCapture(0)

# Canvas ban đầu chưa được tạo vì chưa biết kích thước frame từ camera.
canvas = None

# Lưu tọa độ đầu ngón trỏ ở frame trước để nối thành một nét liên tục.
prev_x, prev_y = None, None

# Chuỗi kết quả nhận dạng để hiển thị lên cửa sổ.
result_text = ""

# Khoảng cách pixel tối đa giữa đầu ngón cái và đầu ngón trỏ để coi là đang chạm.
TOUCH_DISTANCE = 45

# Chuyển canvas BGR thành tensor đúng định dạng đầu vào của mô hình MNIST.
def preprocess_for_model(canvas):
    # OpenCV lưu ảnh màu theo thứ tự BGR; chuyển canvas thành ảnh xám một channel.
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # Pixel > 30 được đặt thành 255, còn lại thành 0 để tách nét trắng khỏi nền đen.
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Lấy tọa độ của tất cả pixel khác 0, tức các pixel thuộc nét chữ.
    coords = cv2.findNonZero(thresh)

    if coords is None:
        return None

    # Tìm hình chữ nhật nhỏ nhất song song với trục bao quanh toàn bộ nét chữ.
    x, y, w, h = cv2.boundingRect(coords)

    # Cắt đúng vùng có chữ số từ ảnh nhị phân.
    digit = thresh[y:y + h, x:x + w]

    # Tạo cạnh vuông bằng chiều lớn hơn giữa width/height và thêm 40 pixel đệm.
    size = max(w, h) + 40

    # Tạo ảnh vuông nền đen có kiểu uint8, tương thích ảnh OpenCV 0-255.
    square = np.zeros((size, size), dtype=np.uint8)

    # Tính vị trí bắt đầu theo chiều ngang để căn giữa chữ số trong ảnh vuông.
    x_offset = (size - w) // 2

    # Tính vị trí bắt đầu theo chiều dọc để căn giữa chữ số trong ảnh vuông.
    y_offset = (size - h) // 2

    # Chép vùng chữ số vào chính giữa ảnh vuông nền đen.
    square[y_offset:y_offset + h, x_offset:x_offset + w] = digit

    # Resize ảnh vuông về 28x28 để khớp kích thước ảnh MNIST dùng lúc train.
    resized = cv2.resize(square, (28, 28))

    # Đổi sang float32 và chuẩn hóa pixel từ [0, 255] về [0, 1].
    resized = resized.astype("float32") / 255.0

    # Thêm chiều batch và channel: (28,28) -> (1,28,28,1).
    resized = resized.reshape(1, 28, 28, 1)

    # Trả tensor đã sẵn sàng cho model.predict().
    return resized

# Khởi tạo MediaPipe Hands bằng context manager để tài nguyên được giải phóng đúng cách.
with mp_hands.Hands(
    # Chỉ phát hiện tối đa một bàn tay để đơn giản hóa logic viết.
    max_num_hands=1,
    # Chỉ chấp nhận phát hiện ban đầu khi confidence đạt ít nhất 0.7.
    min_detection_confidence=0.7,
    # Chỉ tiếp tục theo dõi khi confidence tracking đạt ít nhất 0.7.
    min_tracking_confidence=0.7
) as hands:
    # Vòng lặp vô hạn xử lý liên tục từng frame camera cho tới khi dừng.
    while True:
        # Đọc một frame: ret báo thành công/thất bại, frame chứa ảnh camera.
        ret, frame = cap.read()
        # Nếu không đọc được frame, thoát vòng lặp để tránh xử lý ảnh rỗng.
        if not ret:
            break
        # Lật ngang frame để giao diện hoạt động giống hình ảnh trong gương.
        frame = cv2.flip(frame, 1)
        # Chỉ tạo canvas ở frame đầu tiên, sau khi đã biết shape của camera.
        if canvas is None:
            # Tạo ảnh toàn số 0 có cùng shape và dtype với frame: một nền đen.
            canvas = np.zeros_like(frame)
        # Lấy chiều cao, chiều rộng và số channel của frame.
        h, w, _ = frame.shape
        # OpenCV dùng BGR nhưng MediaPipe yêu cầu RGB, nên phải đổi thứ tự màu.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Chạy phát hiện/theo dõi bàn tay và trả về các landmark đã chuẩn hóa.
        results = hands.process(rgb)
        # Trạng thái mặc định trong frame hiện tại là sẵn sàng.
        status = "SAN SANG"
        # Kiểm tra MediaPipe có tìm thấy ít nhất một bàn tay hay không.
        if results.multi_hand_landmarks:
            # Lấy bàn tay đầu tiên; chương trình đã giới hạn tối đa một bàn tay.
            hand_landmarks = results.multi_hand_landmarks[0]
            # Landmark số 8 là đầu ngón trỏ theo quy ước 21 điểm của MediaPipe.
            index_tip = hand_landmarks.landmark[8]
            # Landmark số 4 là đầu ngón cái theo quy ước của MediaPipe.
            thumb_tip = hand_landmarks.landmark[4]
            # index_tip.x nằm trong khoảng gần [0,1]; nhân w để đổi sang pixel.
            x = int(index_tip.x * w)
            # index_tip.y nằm trong khoảng gần [0,1]; nhân h để đổi sang pixel.
            y = int(index_tip.y * h)
            # Đổi tọa độ x chuẩn hóa của đầu ngón cái sang pixel.
            tx = int(thumb_tip.x * w)
            # Đổi tọa độ y chuẩn hóa của đầu ngón cái sang pixel.
            ty = int(thumb_tip.y * h)

            # Tính khoảng cách Euclid 2D giữa đầu ngón trỏ và đầu ngón cái.
            # Việc bỏ qua trục z có thể làm thao tác chạm kém ổn định theo phối cảnh.
            distance = ((x - tx) ** 2 + (y - ty) ** 2) ** 0.5

            # Nếu hai đầu ngón gần nhau hơn ngưỡng thì coi là lệnh ngắt nét.
            if distance < TOUCH_DISTANCE:
                # Xóa điểm trước để lần viết tiếp theo không nối với nét cũ.
                prev_x, prev_y = None, None

                # Cập nhật trạng thái hiển thị cho người dùng.
                status = "NGAT NET"

                # Vẽ chấm đỏ đặc tại đầu ngón trỏ để báo chế độ ngắt nét.
                cv2.circle(frame, (x, y), 12, (0, 0, 255), -1)

            # Nếu ngón cái và ngón trỏ không chạm nhau thì chuyển sang chế độ viết.
            else:
                # Cập nhật trạng thái hiển thị cho người dùng.
                status = "DANG VIET"

                # Nếu đây là điểm đầu tiên của nét mới, chưa có điểm trước đó.
                if prev_x is None:
                    # Gán điểm hiện tại làm điểm bắt đầu để không nối từ vị trí cũ.
                    prev_x, prev_y = x, y

                # Vẽ đoạn thẳng từ vị trí đầu ngón trỏ cũ đến vị trí hiện tại.
                cv2.line(
                    # Chỉ vẽ lên canvas, không ghi vĩnh viễn lên frame camera.
                    canvas,
                    # Điểm đầu là tọa độ đầu ngón trỏ của frame trước.
                    (prev_x, prev_y),
                    # Điểm cuối là tọa độ đầu ngón trỏ của frame hiện tại.
                    (x, y),
                    # Màu trắng theo thứ tự BGR.
                    (255, 255, 255),
                    # Độ dày nét là 12 pixel.
                    12
                )

                # Lưu vị trí hiện tại để làm điểm đầu cho frame tiếp theo.
                prev_x, prev_y = x, y

                # Vẽ chấm xanh lá lên frame để báo chế độ đang viết.
                cv2.circle(frame, (x, y), 12, (0, 255, 0), -1)

            # Vẽ 21 landmark và các đường nối xương bàn tay lên frame.
            mp_draw.draw_landmarks(
                # Ảnh đích cần vẽ.
                frame,
                # Tập landmark của bàn tay vừa phát hiện.
                hand_landmarks,
                # Danh sách các cặp landmark cần nối với nhau.
                mp_hands.HAND_CONNECTIONS
            )

        # Nhánh này chạy khi frame hiện tại không phát hiện được bàn tay.
        else:
            # Ngắt nét để khi bàn tay xuất hiện lại không có đường nối nhảy vị trí.
            prev_x, prev_y = None, None
            # Cập nhật trạng thái cảnh báo không thấy bàn tay.
            status = "KHONG THAY TAY"

        # Trộn ảnh camera và canvas để vừa thấy người dùng vừa thấy nét đã vẽ.
        # Hai trọng số cộng thành 1.4 nên vùng sáng có thể bị bão hòa về 255.
        combined = cv2.addWeighted(frame, 0.6, canvas, 0.8, 0)

        # Ghi kết quả nhận dạng lên ảnh đã trộn.
        cv2.putText(
            # Ảnh đích để ghi chữ.
            combined,
            # f-string ghép nhãn cố định với kết quả dự đoán hiện tại.
            f"Nhan dien: {result_text}",
            # Tọa độ gốc phía dưới bên trái của dòng chữ.
            (30, 60),
            # Font chữ Hershey Simplex tích hợp sẵn trong OpenCV.
            cv2.FONT_HERSHEY_SIMPLEX,
            # Hệ số phóng đại font.
            1.3,
            # Màu xanh lá theo thứ tự BGR.
            (0, 255, 0),
            # Độ dày nét chữ.
            3
        )

        # Ghi trạng thái thao tác bàn tay lên ảnh.
        cv2.putText(
            # Ảnh đích để ghi chữ.
            combined,
            # Nội dung trạng thái được cập nhật ở các nhánh phía trên.
            status,
            # Vị trí của dòng trạng thái.
            (30, 110),
            # Font chữ OpenCV.
            cv2.FONT_HERSHEY_SIMPLEX,
            # Hệ số phóng đại font.
            1,
            # Màu vàng theo BGR: xanh lá + đỏ.
            (0, 255, 255),
            # Độ dày nét chữ.
            2
        )

        # Ghi hướng dẫn cách ngắt nét ở gần đáy cửa sổ.
        cv2.putText(
            # Ảnh đích để ghi chữ.
            combined,
            # OpenCV putText mặc định không hiển thị Unicode tiếng Việt đầy đủ.
            "Ngon cai cham ngon tro: ngat net",
            # h - 65 giữ dòng chữ cách đáy frame 65 pixel.
            (30, h - 65),
            # Font chữ OpenCV.
            cv2.FONT_HERSHEY_SIMPLEX,
            # Kích thước chữ.
            0.75,
            # Màu trắng theo BGR.
            (255, 255, 255),
            # Độ dày nét chữ.
            2
        )

        # Ghi hướng dẫn các phím điều khiển ở dòng cuối.
        cv2.putText(
            # Ảnh đích để ghi chữ.
            combined,
            # R nhận dạng, C xóa canvas, Q thoát chương trình.
            "R: nhan dien | C: xoa bang | Q: thoat",
            # h - 30 giữ dòng chữ cách đáy frame 30 pixel.
            (30, h - 30),
            # Font chữ OpenCV.
            cv2.FONT_HERSHEY_SIMPLEX,
            # Kích thước chữ.
            0.75,
            # Màu trắng theo BGR.
            (255, 255, 255),
            # Độ dày nét chữ.
            2
        )

        # Hiển thị ảnh kết quả trong cửa sổ có tên "Bang viet bang tay".
        cv2.imshow("Bang viet bang tay", combined)

        # Chờ khoảng 1 ms để OpenCV xử lý sự kiện bàn phím và cửa sổ.
        # & 0xFF lấy 8 bit thấp để chuẩn hóa mã phím trên các hệ điều hành.
        key = cv2.waitKey(1) & 0xFF

        # Nếu người dùng nhấn chữ q thường thì thoát vòng lặp.
        if key == ord("q"):
            break

        # Nếu người dùng nhấn chữ c thường thì xóa bảng viết.
        elif key == ord("c"):
            # Tạo canvas đen mới cùng kích thước frame hiện tại.
            canvas = np.zeros_like(frame)
            # Xóa kết quả nhận dạng cũ khỏi giao diện.
            result_text = ""
            # Ngắt điểm nối để nét mới không nối về vị trí trước khi xóa.
            prev_x, prev_y = None, None

        # Nếu người dùng nhấn chữ r thường thì bắt đầu nhận dạng.
        elif key == ord("r"):
            # Tiền xử lý toàn bộ hình vẽ trên canvas thành tensor cho CNN.
            img = preprocess_for_model(canvas)

            # Chỉ dự đoán khi canvas thật sự có pixel thuộc nét chữ.
            if img is not None:
                # Chạy forward pass; kết quả thường có shape (1, 10).
                # verbose=0 tắt thanh tiến trình của Keras.
                pred = model.predict(img, verbose=0)

                # Lấy chỉ số có xác suất lớn nhất, tương ứng chữ số dự đoán.
                # Không chỉ định axis vẫn đúng ở đây vì batch chỉ chứa một ảnh.
                result_text = str(np.argmax(pred))

            # Nếu hàm tiền xử lý trả None thì canvas chưa có chữ.
            else:
                # Hiển thị thông báo thay vì gọi model với input không hợp lệ.
                result_text = "Chua co chu viet"


# Giải phóng webcam sau khi thoát khỏi vòng lặp.
cap.release()

# Đóng tất cả cửa sổ giao diện do OpenCV tạo ra.
cv2.destroyAllWindows()
