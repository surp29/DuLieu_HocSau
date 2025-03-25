import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# 1. Tải mô hình đã huấn luyện
model = tf.keras.models.load_model('rps_model.h5')
class_names = ['paper', 'rock', 'scissors']

def predict_image(img_path):
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(img_path):
        print(f"Lỗi: Không tìm thấy ảnh '{img_path}'. Vui lòng kiểm tra lại đường dẫn.")
        return

    # 3. Tiền xử lý ảnh
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Chuẩn hóa

    # 4. Dự đoán lớp của ảnh
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # 5. Ánh xạ nhãn dự đoán sang tên lớp
    predicted_label = class_names[predicted_class]

    # 6. Hiển thị ảnh và nhãn dự đoán
    img = Image.open(img_path)
    img.thumbnail((300, 300))  # Thay đổi kích thước ảnh hiển thị
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

    # Chuyển đổi độ tin cậy sang phần trăm
    confidence = np.max(predictions[0]) * 100

    result_label.config(text=f"Dự đoán: {predicted_label}\nĐộ tin cậy: {confidence:.2f}%")

def open_file_dialog():
    filename = filedialog.askopenfilename(initialdir=".",
                                          title="Chọn ảnh",
                                          filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("all files", "*.*")))
    if filename:
        predict_image(filename)

# Tạo cửa sổ Tkinter
root = tk.Tk()
root.title("Rock Paper Scissors - Prediction")

# Tạo các widget
open_button = tk.Button(root, text="Chọn ảnh", command=open_file_dialog)
open_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

root.mainloop()
