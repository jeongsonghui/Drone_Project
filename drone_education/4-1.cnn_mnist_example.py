import numpy as np
import matplotlib.pyplot as plt
from keras import datasets, utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 1. 데이터
print("📂 MNIST 손글씨 데이터 불러오는 중...")
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print(f"훈련용 이미지: {x_train.shape[0]}장, 테스트용 이미지: {x_test.shape[0]}장")

# 2. 전처리
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)
print("✅ 데이터 전처리 완료!")

# 3. 모델
print("🛠 CNN 모델 생성 중...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("✅ 모델 준비 완료!")

# 4. 학습
print("🚀 모델 학습 시작!")
model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=1)

# 5. 평가
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\n📊 테스트 정확도: {accuracy*100:.2f}%")

# 6. 시각화
pred = model.predict(x_test)
num_samples = 5
plt.figure(figsize=(10, 4))
for i in range(num_samples):
    predicted_label = np.argmax(pred[i])
    true_label = np.argmax(y_test[i])
    is_correct = "Correct" if predicted_label == true_label else "Wrong"

    plt.subplot(1, num_samples, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {predicted_label}/ True: {true_label}\n{is_correct}")
    plt.axis("off")
plt.tight_layout()
plt.show()
print("🎯 예측 예시 출력 완료!")
