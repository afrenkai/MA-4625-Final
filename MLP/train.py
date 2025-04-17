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
            batch_x = x_train[i:i+batch_size, :]
            print(batch_x.shape)
            batch_y = y_train[i:i+batch_size]
            dW, dB, loss = back_prop(batch_x, batch_y, W, b, reg_param, relu)
            print(len(dW))
            print(len(dB))

if __name__ == "__main__":
    x_train, y_train = get_data('train')
    print(x_train.shape)
    print(y_train.shape)
    n_feats = x_train.shape[1]
    n_hidden = 64
    n_out = 1
    
    W1 = np.random.randn(n_hidden, n_feats) * 0.01
    b1 = np.zeros((n_hidden))
    W2 = np.random.randn(n_hidden, n_hidden) * 0.01
    b2 = np.zeros(n_hidden)
    W3 = np.random.randn(n_feats, n_hidden) * 0.01
    b3 = np.zeros((1))

    epic = train(x_train, y_train, [W1, W2, W3], [b1, b2, b3])
