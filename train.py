import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Chuẩn bị dữ liệu (đường dẫn đến thư mục dữ liệu)
train_dir = 'train'
validation_dir = 'validation'

# Kiểm tra xem thư mục có tồn tại không
if not os.path.exists(train_dir):
    print(f"Lỗi: Không tìm thấy thư mục '{train_dir}'. Vui lòng tạo thư mục này và đặt dữ liệu vào đó.")
    exit()
if not os.path.exists(validation_dir):
    print(f"Lỗi: Không tìm thấy thư mục '{validation_dir}'. Vui lòng tạo thư mục này và đặt dữ liệu vào đó.")
    exit()

# 2. Tiền xử lý dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

# 3. Xây dựng mô hình
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),  # Thêm lớp Dropout
    layers.Dense(512, activation='relu'),
    layers.Dense(3, activation='softmax') # 3 lớp: rock, paper, scissors
])

# 4. Biên dịch mô hình
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Huấn luyện mô hình
# Sử dụng Callback để lưu lại lịch sử huấn luyện và mô hình
class CustomSaver(tf.keras.callbacks.Callback):
  def __init__(self, filename="model_data.txt"):
    self.filename = filename

  def on_epoch_end(self, epoch, logs=None):
    if logs is None:
      logs = {}
    with open(self.filename, 'a') as f:
        f.write(f"Epoch {epoch+1}: Loss={logs['loss']:.4f}, Accuracy={logs['accuracy']:.4f}, Validation Loss={logs['val_loss']:.4f}, Validation Accuracy={logs['val_accuracy']:.4f}\n")

# Tạo instance của callback
custom_saver = CustomSaver()

history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
    callbacks=[custom_saver]  # Truyền callback vào hàm fit
)

# 6. Đánh giá mô hình
loss, accuracy = model.evaluate(validation_generator)
print('Độ chính xác trên tập kiểm tra: {:.2f}%'.format(accuracy * 100))

# 7. Lưu mô hình
model.save('rps_model.h5')

# Hiển thị kết quả huấn luyện
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
