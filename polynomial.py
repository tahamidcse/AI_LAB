
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
def main(n):
    #--- Build model
    model = build_model()
    model.compile(loss = 'mse')
    
    #--- Prepare data
    (trainX, trainY), (valX, valY), (testX, testY),maxi,mini = prepare_train_val_test(n)
    
    # --- The data needs to be reshaped to (number of samples, 1) to match the model's input layer
    trainX = trainX.reshape(-1, 1)
    trainY = trainY.reshape(-1, 1)
    valX = valX.reshape(-1, 1)
    valY = valY.reshape(-1, 1)
    testX = testX.reshape(-1, 1)
    testY = testY.reshape(-1, 1)
    #--- Train model
    print(f"\n--- Training with n = {n} ---")
    if n==1000:
      e=100
    elif n==500:
      e=75
    else:
      e=50
    model.fit(trainX, trainY, validation_data = (valX, valY), epochs = e)
    plot_results(model, n, testX, testY, maxi, mini)

def normalize(values):
    maximum = np.max(values)
    minimum = np.min(values)
    return ((2 * (values - minimum) / (maximum - minimum)) - 1),maximum,minimum 

def denormalize(norm, maxi, mini):
   
    return ((norm + 1) / 2) * (maxi - mini) + mini

def prepare_train_val_test(n_value):
    x, y,maxi,mini = data_process(n_value)
    total_n = len(x)
    print(f"\nShape of x: {x.shape}, Total samples: {total_n}")
    
    combined_data = np.c_[x, y]
    np.random.shuffle(combined_data)

    x = combined_data[:, 0]
    y = combined_data[:, 1]
    
    train_n = int(total_n * 0.7)
    val_n = int(total_n * 0.1)
    test_n = int(total_n * 0.2)
    
    trainX = x[: train_n]
    trainY = y[: train_n]
    valX = x[train_n : train_n + val_n]
    valY = y[train_n : train_n + val_n]
    testX = x[train_n + val_n :]
    testY = y[train_n + val_n :]
    print(f'Total samples: {len(x)}, Training samples: {len(trainX)}, Validation samples: {len(valX)}, Test samples: {len(testX)}')
    
    return (trainX, trainY), (valX, valY), (testX, testY),maxi,mini

def data_process(n):
    y = []
    x = np.random.randint(0, n, n)
    for i in range(n):
        y.append(my_polynomial(x[i]))
    y = np.array(y)
    x = np.array(x)
 
    x,a,b = normalize(x)
    y,maxi,mini = normalize(y)
    
    return x, y,maxi,mini
    
def my_polynomial(x):
    y = 5 * x**2 + 10 * x -5
    return y

def build_model():
    inputs = Input((1,))
    h2 = Dense(32, activation='relu', name='h2')(inputs)
    h3 = Dense(64, activation='relu', name='h3')(h2)
    h4 = Dense(128, activation='relu', name='h4')(h3)
    outputs = Dense(1, name = 'output_layer')(h4)

    model = Model(inputs, outputs)
    model.summary(show_trainable = True)
    
    return model
def plot_results(model, n_value, testX, testY, y_max, y_min):
  
    # Predict values on the test data
    py_norm = model.predict(testX)
    
    # Denormalize the data for plotting
    predY = denormalize(py_norm, y_max, y_min)
    testY_denorm = denormalize(testY, y_max, y_min)
    testX_denorm = denormalize(testX, np.max(testX), np.min(testX))

    # Sort the data based on the x-values for a clean plot
    srt_ind = np.argsort(testX_denorm.flatten())
    srt_tx = testX_denorm[srt_ind]
    srt_ty = testY_denorm[srt_ind]
    srt_py = predY[srt_ind]
    
    plt.figure(figsize=(10, 6))
    plt.plot(srt_tx, srt_ty, 'b-', label='Original Curve')
    plt.plot(srt_tx, srt_py, 'r--', label='Predicted Curve')
    plt.title(f'Original vs. Predicted Curve (n = {n_value})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    for n in [100, 500, 1000]:
        main(n)
