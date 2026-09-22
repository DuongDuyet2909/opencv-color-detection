import tensorflow as tf
from tensorflow.keras import layers, models
# Đọc dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Chuyển đổi dữ liệu sang định dạng phù hợp cho mô hình CNN
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
# Kiến trúc CNN
model = models.Sequential([
    layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])
# Cấu hình mô hình
model.compile(
    optimizer="adam", # Adam là thuật toán cập nhật trọng số dựa trên gradient, có learning rate thích nghi cho từng tham số.
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
# Huấn luyện mô hình
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
model.save("digit_model.h5")

print("Đã lưu model: digit_model.h5")