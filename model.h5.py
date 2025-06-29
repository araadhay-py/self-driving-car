from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Lambda

model = Sequential([
    Lambda(lambda x: x / 255.0, input_shape=(66, 200, 3)),
    Conv2D(24, (5, 5), strides=(2, 2), activation="relu"),
    Conv2D(36, (5, 5), strides=(2, 2), activation="relu"),
    Conv2D(48, (5, 5), strides=(2, 2), activation="relu"),
    Conv2D(64, (3, 3), activation="relu"),
    Conv2D(64, (3, 3), activation="relu"),
    Flatten(),
    Dense(100, activation="relu"),
    Dense(50, activation="relu"),
    Dense(10, activation="relu"),
    Dense(1)  # output = steering angle
])
