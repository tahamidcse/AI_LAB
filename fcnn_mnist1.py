from tensorflow.keras.datasets.mnist import load_data
import numpy as np
import random
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model


def build_model():
    number_of_classes = 10
    inputs = Input((28, 28, 1), name='Input Layer')
    h1 = Flatten()(inputs)
    h2 = Dense(32, activation='relu', name='h2')(h1)
    h3 = Dense(64, activation='relu', name='h3')(h2)
    h4 = Dense(128, activation='relu', name='h4')(h3)
    outputs = Dense(number_of_classes, activation='softmax', name='output_layer')(h4)
    model = Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    # --- Build model
    model = build_model()

    # --- Load and preprocess data
    (trainX, trainY), (testX, testY) = load_data()
    trainX = trainX.reshape(-1, 28, 28, 1).astype("float32") / 255.0)
    testX = testX.reshape(-1, 28, 28, 1).astype("float32") / 255.0)

# Convert to binary labels: 0 for even, 1 for odd
def is_odd(y):
    return y % 2

trainY_binary = np.array([is_odd(y) for y in trainY])
testY_binary = np.array([is_odd(y) for y in testY])


    trainY = to_categorical(trainY, num_classes=10)
    testY = to_categorical(testY, num_classes=10)

    # --- Train model
    model.fit(trainX, trainY, epochs=1, batch_size=32, validation_data=(testX, testY))

    # --- Show model summary
    model.summary()

    # --- Evaluate model
    loss, accuracy = model.evaluate(testX, testY, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")

    # --- Prediction and comparison
    predict = model.predict(testX)
    predicLabel = np.argmax(predict, axis=1)
    trueLabel = np.argmax(testY, axis=1)
    print('True Labels: {}\nPredicted Labels: {}\n'.format(trueLabel, predicLabel))

    # --- Display 25 random test images with predictions
    K = 1  # Number of times to show a batch of 25 images (set >1 if needed)
    for _ in range(K):
        indices = random.sample(range(testX.shape[0]), 25)
        fig = plt.figure('Digits of Test Set', figsize=(20, 20))
        for j, i in enumerate(indices, 1):
            fig.add_subplot(5, 5, j)
            title = f'[TL: {trueLabel[i]}] [PL: {predicLabel[i]}]'
            plt.title(title)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(testX[i, :, :, 0], cmap='gray')
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
