from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
import cv2

# --- Build model
inputs = Input((28, 28, 1), name='Input Layer')
h1 = Flatten()(inputs)
h2 = Dense(32, activation='relu', name='h2')(h1)
h3 = Dense(64, activation='relu', name='h3')(h2)
h4 = Dense(128, activation='relu', name='h4')(h3)
outputs = Dense(10, activation='softmax', name='output_layer')(h4)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Load MNIST data
(trainX, trainY), (testX, testY) = mnist.load_data()

# --- Load and preprocess your custom image
img_path = '/content/sample_data/sample_image(1).png'
custom_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)   # grayscale
custom_img = cv2.resize(custom_img, (28, 28))             # resize
custom_img = custom_img.astype("float32") / 255.0         # normalize
custom_img = np.expand_dims(custom_img, axis=-1)          # add channel dim

# --- Label for your custom image (set manually)
custom_label = 5   # 👈 change this to the correct digit for your image

# --- Add custom image to training set
trainX = np.concatenate([trainX, np.expand_dims(custom_img, axis=0)], axis=0)
trainY = np.concatenate([trainY, [custom_label]], axis=0)

# --- Preprocess datasets
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0
trainX = trainX.reshape(-1, 28, 28, 1)
testX = testX.reshape(-1, 28, 28, 1)
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# --- Train model
model.fit(trainX, trainY, epochs=5, batch_size=32, validation_split=0.2)

# --- Evaluate on MNIST test set
test_loss, test_acc = model.evaluate(testX, testY)
print(f"FCNN test accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# --- Now test on your custom image
img_for_test = np.expand_dims(custom_img, axis=0)   # batch dim
prediction = model.predict(img_for_test)
predicted_class = np.argmax(prediction)

print(f"Predicted class for custom image: {predicted_class}")

# --- Show image
plt.imshow(custom_img[:, :, 0], cmap='gray')
plt.title(f"Predicted: {predicted_class}, True: {custom_label}")
plt.show()
