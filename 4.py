import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist, mnist, cifar10
import matplotlib.pyplot as plt
import numpy as np


# ==============================
# Build Fully Connected Network
# ==============================
def build_fcf_nn(input_shape, num_classes):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
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

    num_classes = len(np.unique(trainY))

    trainY = to_categorical(trainY, num_classes)
    testY = to_categorical(testY, num_classes)

    model = build_fcf_nn(trainX.shape[1:], num_classes)
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

    # Loss Plot
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(dataset_name + " Loss")
    plt.legend(['Train', 'Validation'])
    plt.savefig(dataset_name + "_loss.png")
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
