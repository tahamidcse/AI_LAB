import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist, mnist, cifar10
import numpy as np
import matplotlib.pyplot as plt


# ==============================
# Build CNN Model
# ==============================
def build_cnn(input_shape, num_classes):

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),

        Conv2D(128, (3,3), activation='relu'),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ==============================
# Train and Evaluate
# ==============================
def train_and_evaluate(dataset_name, load_function):

    print(f"\n========== {dataset_name} ==========\n")

    (trainX, trainY), (testX, testY) = load_function()

    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0

    # Add channel dimension for grayscale datasets
    if len(trainX.shape) == 3:
        trainX = np.expand_dims(trainX, axis=-1)
        testX = np.expand_dims(testX, axis=-1)

    num_classes = len(np.unique(trainY))

    trainY = to_categorical(trainY, num_classes)
    testY = to_categorical(testY, num_classes)

    model = build_cnn(trainX.shape[1:], num_classes)
    model.summary()

    history = model.fit(
        trainX, trainY,
        validation_split=0.1,
        epochs=20,
        batch_size=64,
        verbose=1
    )

    loss, accuracy = model.evaluate(testX, testY, verbose=0)
    print(f"Test Accuracy for {dataset_name}: {accuracy:.4f}")

    # Accuracy Plot
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(dataset_name + " Accuracy")
    plt.legend(['Train', 'Validation'])
    plt.savefig(dataset_name + "_accuracy.png")
    plt.close()

    return accuracy


# ==============================
# Main
# ==============================
if __name__ == "__main__":

    acc_fashion = train_and_evaluate("Fashion_MNIST", fashion_mnist.load_data)

    acc_mnist = train_and_evaluate("MNIST", mnist.load_data)

    acc_cifar = train_and_evaluate("CIFAR10", cifar10.load_data)

    print("\nFinal Test Accuracies:")
    print("Fashion MNIST:", acc_fashion)
    print("MNIST:", acc_mnist)
    print("CIFAR-10:", acc_cifar)
