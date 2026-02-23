# for ease of use we used numbers .0 represents siuli 1 represents togor.
#======================= Necessary Imports =========================
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16
from tensorflow.keras.utils import to_categorical
import cv2
import random
def main():
    #--- Load custom dataset (flower1 and flower2)
    custom = np.load('flower.npz')
    custom_trainX = custom['trainX'].astype('float32') / 255.0
    custom_trainY = custom['trainY'].astype('int')
    custom_testX = custom['testX'].astype('float32') / 255.0
    custom_testY = custom['testY'].astype('int')

    #--- Convert labels to one-hot
    custom_trainY = to_categorical(custom_trainY, 2)
    custom_testY = to_categorical(custom_testY, 2)

    print("Custom Train Shape :", custom_trainX.shape, custom_trainY.shape)
    print("Custom Test Shape  :", custom_testX.shape, custom_testY.shape)

    #--- Build model
    model = build_model()

    #--- Compile and train
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(custom_trainX, custom_trainY, validation_split=0.1, epochs=10, batch_size=32)

    #--- Evaluate
    test_loss, test_acc = model.evaluate(custom_testX, custom_testY, verbose=0)
    print("Custom Test Accuracy:" ,test_acc)

    #--- Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Custom flower Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Custom flower Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    predict = model.predict(custom_testX)
    predicLabel = np.argmax(predict, axis=1)
    trueLabel = custom_testY
    print('True Labels: {}\nPredicted Labels: {}\n'.format(trueLabel, predicLabel))



    K = 1  # Number of times to show a batch of 25 images (set >1 if needed)
    for _ in range(K):
        indices = random.sample(range(custom_testX.shape[0]), 25)
        fig = plt.figure('flowers of Test Set', figsize=(20, 20))
        for j, i in enumerate(indices, 1):
            fig.add_subplot(5, 5, j)
            title = f'[TL: {trueLabel[i]}] [PL: {predicLabel[i]}]'
            plt.title(title)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(custom_testX[i, :, :,:])
        plt.tight_layout()
        plt.show()
    plt.close()


#======================= Model Construction =========================show image
def build_model():
    # Load VGG16 without top, with pretrained weights
    base_model = vgg16.VGG16(input_shape=(32, 32, 3), weights='imagenet', include_top=False)

    for layer in base_model.layers:
        layer.trainable = False  # Freeze base model

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)#sigmoid

    model = Model(inputs=base_model.input, outputs=outputs)
    return model

#======================= Entry Point =========================
if __name__ == '__main__':
    main()
