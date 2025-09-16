

from tensorflow.keras.datasets.fashion_mnist import load_data
import numpy as np
import random
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
def build_model():
    number_of_classes=10
    inputs = Input((28,28,1),name='Input Layer')
    

    x = Conv2D(filters = 8, kernel_size = (5, 5), padding = 'same', activation = 'relu')(inputs)
    x = Conv2D(filters = 8, kernel_size = (5, 5), padding = 'same', activation = 'relu')(    x)
    x = MaxPooling2D()(    x) # Downsampling
    x = Conv2D(filters = 16, kernel_size = (5, 5), activation = 'relu')(    x)
    x = Conv2D(filters = 16, kernel_size = (5, 5), activation = 'relu')(    x)
    x = Conv2D(filters = 16, kernel_size = (5, 5), strides = (2, 2), activation = 'relu')(    x) # Downsampling
    x = Flatten()(    x)
    x = Dense(8, activation = 'relu')(    x)
    outputs = Dense(number_of_classes,activation='softmax', name = 'output_layer')(    x)
    model = Model(inputs, outputs, name = 'CNN')
   

    model.compile(optimizer="rmsprop", loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model



def main():
    # --- Build model
    model = build_model()

    # --- Load and preprocess data
    (trainX, trainY), (testX, testY) = load_data()
    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0
    trainX = np.expand_dims(trainX, -1)
    testX = np.expand_dims(testX, -1)

    trainY = to_categorical(trainY, num_classes=10)
    testY = to_categorical(testY, num_classes=10)

    # --- Train model
    model.fit(trainX, trainY, epochs=5, batch_size=32, validation_data=(testX, testY))

    # --- Show model summary
    model.summary()

    # --- Evaluate model
    loss, accuracy = model.evaluate(testX, testY, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")

    # --- Prediction and comparison
    predict = model.predict(testX)
    predicLabel = np.argmax(predict, axis=1)
    trueLabel = np.argmax(testY, axis=1)
    print("Sample Predictions:")
    for i in range(10):
        print(f"True: {true_labels[i]}  Predicted: {predicted_labels[i]}")

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

    
