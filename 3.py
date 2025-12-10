import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# Select which equation to learn
# ----------------------------------------------------------------
def my_polynomial(x, mode):
    if mode == 1:        # Linear
        return 5*x + 10
    elif mode == 2:      # Quadratic
        return 3*x**2 + 5*x + 10
    elif mode == 3:      # Cubic
        return 4*x**3 + 3*x**2 + 5*x + 10


# ----------------------------------------------------------------
# MAIN EXECUTION FUNCTION
# ----------------------------------------------------------------
def main(n, mode):
    print("\n==============================")
    print(f"Training FCFNN for Equation {mode}, n = {n}")
    print("==============================")

    #--- Build model
    model = build_model()
    model.compile(loss='mse', optimizer='adam')

    #--- Prepare data
    (trainX, trainY), (valX, valY), (testX, testY), ymax, ymin = prepare_train_val_test(n, mode)

    # reshape for Keras
    trainX = trainX.reshape(-1, 1)
    trainY = trainY.reshape(-1, 1)
    valX = valX.reshape(-1, 1)
    valY = valY.reshape(-1, 1)
    testX = testX.reshape(-1, 1)
    testY = testY.reshape(-1, 1)

    #--- Training
    epochs = 100 if n == 1000 else (75 if n == 500 else 50)
    model.fit(trainX, trainY, validation_data=(valX, valY), epochs=epochs, verbose=0)

    #--- Plot results
    plot_results(model, n, mode, testX, testY, ymax, ymin)


# ----------------------------------------------------------------
# NORMALIZATION
# ----------------------------------------------------------------
def normalize(values):
    maximum = np.max(values)
    minimum = np.min(values)
    return ((2 * (values - minimum) / (maximum - minimum)) - 1), maximum, minimum

def denormalize(norm, maximum, minimum):
    return ((norm + 1) / 2) * (maximum - minimum) + minimum


# ----------------------------------------------------------------
# DATA PREPARATION
# ----------------------------------------------------------------
def prepare_train_val_test(n_value, mode):
    x, y, ymax, ymin = data_process(n_value, mode)
    total_n = len(x)
    combined_data = np.c_[x, y]
    np.random.shuffle(combined_data)

    x = combined_data[:, 0]
    y = combined_data[:, 1]

    # 70/10/20 split
    train_n = int(total_n * 0.7)
    val_n = int(total_n * 0.1)

    trainX = x[:train_n]
    trainY = y[:train_n]
    valX = x[train_n: train_n + val_n]
    valY = y[train_n: train_n + val_n]
    testX = x[train_n + val_n:]
    testY = y[train_n + val_n:]

    return (trainX, trainY), (valX, valY), (testX, testY), ymax, ymin


# ----------------------------------------------------------------
# DATA GENERATION
# ----------------------------------------------------------------
def data_process(n, mode):
    x = np.random.randint(0, n, n)
    y = np.array([my_polynomial(val, mode) for val in x])

    x, xmax, xmin = normalize(x)
    y, ymax, ymin = normalize(y)

    return x, y, ymax, ymin


# ----------------------------------------------------------------
# MODEL ARCHITECTURE
# ----------------------------------------------------------------
def build_model():
    inputs = Input((1,))
    h1 = Dense(32, activation='relu')(inputs)
    h2 = Dense(64, activation='relu')(h1)
    h3 = Dense(128, activation='relu')(h2)
    outputs = Dense(1)(h3)
    return Model(inputs, outputs)


# ----------------------------------------------------------------
# PLOTTING (Original vs Predicted)
# ----------------------------------------------------------------
def plot_results(model, n_value, mode, testX, testY, y_max, y_min):
    pred_norm = model.predict(testX)
    predY = denormalize(pred_norm, y_max, y_min)
    trueY = denormalize(testY, y_max, y_min)
    testX_denorm = denormalize(testX, np.max(testX), np.min(testX))

    # Sort for cleaner curve
    srt = np.argsort(testX_denorm.flatten())
    sx = testX_denorm[srt]
    sy_true = trueY[srt]
    sy_pred = predY[srt]

    titles = {
        1: "Linear: y = 5x + 10",
        2: "Quadratic: y = 3x² + 5x + 10",
        3: "Cubic: y = 4x³ + 3x² + 5x + 10"
    }

    plt.figure(figsize=(8, 6))
    plt.scatter(sx, sy_true, color="blue", s=15, label="Original y")
    plt.scatter(sx, sy_pred, color="red", s=15, label="Predicted y")
    plt.title(f"{titles[mode]}  (n={n_value})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


# ----------------------------------------------------------------
# RUN ALL THREE EQUATIONS
# ----------------------------------------------------------------
if __name__ == "__main__":
    for mode in [1, 2, 3]:     # Linear, Quadratic, Cubic
        for n in [100, 500, 1000]:
            main(n, mode)
