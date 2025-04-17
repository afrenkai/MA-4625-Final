from arch import back_prop, relu
import numpy as np
import pandas as pd
from tqdm import tqdm
from mlp_utils import get_data

def train(x_train: np.ndarray, y_train: np.ndarray, W: list[np.ndarray], b: list[np.ndarray], 
          learning_rate=1e-4, batch_size=32, num_epochs=100, reg_param=1e-4, clip_val=1.0, act_fn=relu):
    losses = []
    # print(f"Number of weight matrices: {len(W)}")
    
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        # Create random permutation of samples
        idx = np.random.permutation(x_train.shape[0])  # Should be shuffling along samples, not features
        x_train_shuffled, y_train_shuffled = x_train[idx], y_train[idx]
        
        for i in range(0, x_train.shape[0], batch_size):  # Iterate over samples
            batch_x = x_train_shuffled[i:i+batch_size]  # Extract batch of samples
            batch_y = y_train_shuffled[i:i+batch_size]
            
            # Debug prints
            # print(f"Batch x shape: {batch_x.shape}")
            
            # Call backprop
            dW, dB, loss = back_prop(batch_x, batch_y, W, b, reg_param, act_fn)
            
            # Debug prints
            # print(f"Number of dW matrices: {len(dW)}")
            # print(f"Number of dB vectors: {len(dB)}")
            
            # Update weights (this part was missing in your code)
            for j in range(len(W)):
                W[j] -= learning_rate * np.clip(dW[j], -clip_val, clip_val)
                b[j] -= learning_rate * np.clip(dB[j], -clip_val, clip_val)
            
            epoch_loss += loss
        
        avg_epoch_loss = epoch_loss / (x_train.shape[0] // batch_size + 1)
        losses.append(avg_epoch_loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {avg_epoch_loss}")
    
    return losses

if __name__ == "__main__":  # Fixed the name check
    x_train, y_train = get_data('train')
    # print(f"x_train shape: {x_train.shape}")
    # print(f"y_train shape: {y_train.shape}")
    
    n_feats = x_train.shape[1]
    n_hidden = 64
    n_out = 1
    
    # Initialize weights
    W1 = np.random.randn(n_hidden, n_feats) * 0.01
    b1 = np.zeros(n_hidden)
    W2 = np.random.randn(n_hidden, n_hidden) * 0.01
    b2 = np.zeros(n_hidden)
    W3 = np.random.randn(n_out, n_hidden) * 0.01  # Changed to n_out for output layer
    b3 = np.zeros(n_out)
    
    losses = train(x_train, y_train, [W1, W2, W3], [b1, b2, b3])
    print(losses)
