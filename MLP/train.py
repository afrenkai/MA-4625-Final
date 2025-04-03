from arch import back_prop, relu
import numpy as np
import pandas as pd
from tqdm import tqdm
from mlp_utils import get_data

def train(x_train: np.ndarray, y_train: np.ndarray, W: list[np.ndarray], b: list[np.ndarray], learning_rate= 1e-4, batch_size=32, num_epochs = 1000, reg_param = 1e-4, clip_val = 1.0, act_fn = relu):
    losses = []
    print(len(W))
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        idx = np.random.permutation(x_train.shape[1])
        x_train, y_train = x_train[:, idx], y_train[idx]
        for i in range(0, x_train.shape[1], batch_size):
            batch_x = x_train[:, i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            dW, dB, loss = back_prop(batch_x, batch_y, W, b, reg_param, relu)
            print(dW)
            print(dB)

if __name__ == "__main__":
    x_train, y_train = get_data('train')
    print(x_train.shape)
    print(y_train.shape)
    W1 = np.random.rand(4, x_train.shape[-1])
    b1 = np.random.rand(4)
    W2 = np.random.rand(3, 4)
    b2 = np.random.rand(3)
    W3 = np.random.rand(2, 3)
    b3 = np.random.rand(2)
    W4 = np.random.rand(1, 2)
    b4 = np.random.rand(1)
    W = [W1, W2, W3, W4]
    b =  [b1, b2, b3, b4]
    epic = train(x_train, y_train, W, b)
