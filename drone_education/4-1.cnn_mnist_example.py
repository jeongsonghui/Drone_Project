import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 1. 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 데이터 전처리
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. CNN 모델 만들기
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 4. 컴파일 및 훈련
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Training model...")
model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=1)

# 5. 평가
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ 테스트 정확도: {accuracy:.4f}")

# 6. 예측 시각화 (첫 번째 테스트 이미지)
pred = model.predict(x_test)
predicted_label = np.argmax(pred[0])
true_label = np.argmax(y_test[0])

plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f"예측: {predicted_label}, 실제: {true_label}")
plt.axis("off")
plt.show()
