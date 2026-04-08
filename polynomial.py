import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import metrics
import tensorflow as tf

def calculate_y(x):
    return 5*(x**3) - 10*(x**2) - 20*x + 10

def dnn_model():
    input_layer = Input((1,), name='InputLayer')
    hidden_layer1 = Dense(32, activation='relu')(input_layer)
    hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)
    hidden_layer3 = Dense(128, activation='relu')(hidden_layer2)
    output_layer = Dense(1, name='OutputLayer')(hidden_layer3)

    model = Model(inputs=input_layer, outputs=output_layer, name="DNN_Model")
    return model

def generate_samples(n):
    x = np.random.uniform(-20, 20, n)
    y = calculate_y(x)
    return x, y

def normalize(data):
    max_val = np.max(data)
    min_val = np.min(data)
    return 2 * (data - min_val) / (max_val - min_val) - 1

def main():
    model = dnn_model()
    model.summary()

    n = 5000
    x, y = generate_samples(n)

    x_norm = normalize(x)
    y_norm = normalize(y)

    x_train = x_norm[:int(0.90 * n)]
    y_train = y_norm[:int(0.90 * n)]

    x_val = x_norm[int(0.90 * n):int(0.95 * n)]
    y_val = y_norm[int(0.90 * n):int(0.95 * n)]

    x_test = x_norm[int(0.95 * n):]
    y_test = y_norm[int(0.95 * n):]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=[
            metrics.MeanSquaredError(name='mse'),
            metrics.MeanAbsoluteError(name='mae'),
            metrics.R2Score(name='r2')
        ]
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=50,
        validation_data=(x_val, y_val),
        verbose=1
    )

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Training vs Validation MAE')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history.history['r2'], label='Training Accuracy (R²)')
    plt.plot(history.history['val_r2'], label='Validation Accuracy (R²)')
    plt.xlabel('Epochs')
    plt.ylabel('R² Score')
    plt.title('Training vs Validation Accuracy (R²)')
    plt.legend()
    plt.grid(True)

    y_pred = model.predict(x_test)

    plt.subplot(2, 2, 4)
    plt.scatter(x_test, y_test, label='Ground Truth', alpha=0.9)
    plt.scatter(x_test, y_pred, label='Prediction', alpha=0.5)
    plt.xlabel('x (normalized)')
    plt.ylabel('y (normalized)')
    plt.title('Ground Truth vs Prediction')
    plt.legend()
    plt.grid(True)
    test_loss, test_mse, test_mae, test_r2 = model.evaluate(x_test, y_test)
    print("Test Loss (MSE):", test_loss)
    print("Test MSE:", test_mse)
    print("Test MAE:", test_mae)
    print("Test Accuracy (R²):", test_r2)
    plt.tight_layout()
    plt.show()

   

if name == "main":
    main()
