import numpy as np

def relu(z: np.ndarray):
    return np.maximum(0, z)

def sigmoid(z: np.ndarray):     
    return 1 / (1 + np.exp(-z))

def tanh(z: np.ndarray | int | float):
    res = ((np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z)))
    assert np.abs(res - np.tanh(z)) <= 1e-5
    return res

def half_mse(yhat, y, W: list[np.ndarray], alpha):
    # Ensure y and yhat have compatible shapes
    y = y.reshape(-1, 1)
    yhat = yhat.reshape(-1, 1)
    
    loss = 0.5 * np.mean((y - yhat) ** 2)
    reg = (alpha/2)
    reg_param = 0
    for w in W:
        reg_param += (np.sum(w ** 2))
    reg *= reg_param
    return loss + reg

def forward_prop(x, y, W: list[np.ndarray], b: list[np.ndarray], alpha, act_fn):
    if len(W) != len(b):
        print('unequal amount of weights and biases')
        return None
    
    # Ensure x is shaped correctly for matrix multiplication (samples, features)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)  # Single sample case
    
    # For multiple samples, we want shape (features, samples)
    x_col = x.T
    # print(f'x col shape: {x_col.shape}')
    # print(f'w[0] shape: {W[0].shape}')
    
    z = {}
    h = {}
    
    # First layer forward pass
    z[0] = W[0] @ x_col + b[0][:, np.newaxis]
    h[0] = act_fn(z[0])
    
    # Forward through remaining layers
    for w in range(1, len(W)):
        # print(f'Weight at index {w}: {W[w].shape}')
        # print(f'activation at index {w-1}: {h[w-1].shape}')
        # print(f'bias at index {w}: {b[w].shape}')
        
        z[w] = W[w] @ h[w-1] + b[w][:, np.newaxis]
        
        # Apply activation function except for output layer (unless sigmoid)
        if w < len(W) - 1 or act_fn == sigmoid:
            h[w] = act_fn(z[w])
        else:
            h[w] = z[w]
    
    # Debug output
    # for k, v in z.items():
        # print(f'z layer {k}: {v.shape}')
    # for k, v in h.items():
        # print(f'h layer {k}: {v.shape}')
    
    # Calculate prediction and loss
    y_hat = h[len(W) - 1] if len(W) - 1 in h else z[len(W) - 1]
    # print(f'y_hat shape: {y_hat.shape}')
    
    # Ensure y is properly shaped for loss calculation
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    loss = half_mse(y_hat, y, W, alpha)
    
    return loss, z, h, x, y_hat

def back_prop(X, y, W: list[np.ndarray], B: list[np.ndarray], alpha, act_fn):
    # Ensure X has correct shape
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    # print("Starting forward propagation...")
    loss, z, h, x, yhat = forward_prop(X, y, W, B, alpha, act_fn)
    # print(f'loss from fprop: {loss}')
    
    # Ensure y is shaped correctly
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    # Initialize gradient lists
    dW = []
    dB = []
    n_layers = len(W)
    
    # Backpropagation
    delta = None
    for i in range(n_layers - 1, -1, -1):
        # Output layer error
        if i == n_layers - 1:
            # For MSE loss: delta = -(y - yhat)
            # Note: The negative sign because we minimize loss (gradient descent)
            delta = -(y - yhat.T).T
        else:
            # Hidden layer error depends on activation function
            if act_fn == relu:
                delta = (W[i+1].T @ delta) * (z[i] > 0)
            elif act_fn == sigmoid:
                delta = (W[i+1].T @ delta) * (h[i] * (1 - h[i]))
            elif act_fn == tanh:
                delta = (W[i+1].T @ delta) * (1 - h[i]**2)
            else:
                print('Unsupported activation function')
                return None
        
        # Compute gradients
        if i > 0:
            # For hidden layers, use activations from the previous layer
            dW_i = delta @ h[i-1].T / X.shape[0]
        else:
            # For first layer, use input data
            dW_i = delta @ X / X.shape[0]
        
        # Bias gradient is just the delta, averaged over batch size
        dB_i = np.mean(delta, axis=1)
        
        # Add L2 regularization gradient for weights
        dW_i = dW_i + alpha * W[i]
        
        # Store gradients
        dW.insert(0, dW_i)
        dB.insert(0, dB_i)
    
    return dW, dB, loss

def train(x_train: np.ndarray, y_train: np.ndarray, W: list[np.ndarray], b: list[np.ndarray], 
          learning_rate=1e-4, batch_size=32, num_epochs=1000, reg_param=1e-4, clip_val=1.0, act_fn=relu):
    losses = []
    # print(f"Number of weight matrices: {len(W)}")
    
    # Ensure input data has proper shape
    if len(x_train.shape) == 1:
        x_train = x_train.reshape(1, -1)
    
    # Make sure y_train is a vector
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        raise ValueError("y_train should be a vector")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        # Create random permutation of samples
        idx = np.random.permutation(x_train.shape[0])
        x_train_shuffled, y_train_shuffled = x_train[idx], y_train[idx]
        
        num_batches = 0
        for i in range(0, x_train.shape[0], batch_size):
            end_idx = min(i + batch_size, x_train.shape[0])
            batch_x = x_train_shuffled[i:end_idx]
            batch_y = y_train_shuffled[i:end_idx]
            
            # Debug prints
            # print(f"Batch x shape: {batch_x.shape}")
            # print(f"Batch y shape: {batch_y.shape}")
            
            # Call backprop
            dW, dB, loss = back_prop(batch_x, batch_y, W, b, reg_param, act_fn)
            
            # Update weights
            for j in range(len(W)):
                W[j] -= learning_rate * np.clip(dW[j], -clip_val, clip_val)
                b[j] -= learning_rate * np.clip(dB[j], -clip_val, clip_val)
            
            epoch_loss += loss
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        losses.append(avg_epoch_loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {avg_epoch_loss}")
    
    return losses

if __name__ == "__main__":
    # Test with a simple example
    x = np.array([[1, 2], [3, 4]])  # 2 samples with 2 features each
    y = np.array([0, 1])            # 2 labels
    
    # Initialize weights for a 2-layer network
    n_features = x.shape[1]
    n_hidden = 4
    n_out = 1
    
    W1 = np.random.randn(n_hidden, n_features) * 0.01
    b1 = np.zeros(n_hidden)
    W2 = np.random.randn(n_out, n_hidden) * 0.01
    b2 = np.zeros(n_out)
    
    # Test backprop
    gradients_w, gradients_b, loss = back_prop(x, y, [W1, W2], [b1, b2], 0.1, relu)
    
    print("Test completed successfully!")
    # for w in range(len(gradients_w)):
        # print(f'Layer {w+1} weight gradient shape: {gradients_w[w].shape}'
