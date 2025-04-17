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
    yhat = yhat.reshape(-1, 1)
    loss = 0.5 * np.mean((y-yhat) ** 2)
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
    b_0 = b[0]
    x_col = x.T
    print(f'x col shape: {x_col.shape}')
    print(f'w[0] shape: {W[0].shape}')
    z = {}
    h = {}
    z[0] = W[0] @ x_col + b_0[:, np.newaxis]
    h[0] = act_fn(z[0])


    loss = 0
    y_hat = None
    
    if len(W) >  2:
        for w in range(1 , len(W)):
            print(f' Weight at index {w}: {W[w].shape}')
            print(f'score at index {w}: {h[w-1].shape}')
            print(f'bias at index {w}: {b[w].shape}')
            z[w] = W[w] @ z[w-1] + b[w][:, np.newaxis]
            if w < len(W) - 1 or act_fn == sigmoid:

            # print(z[w])
                h[w] = act_fn(z[w])
            else:
                h[w] = z[w]
        for k, v in h.items():
            print(f'z layer {k}: {v.shape}')
        for k, v in z.items():
            print(f'h layer{k}: {v.shape}')

        y_hat = z[len(W) - 1]
        print(y_hat.shape)
        loss = (half_mse(y_hat, y, W, alpha))
    elif len(W) == 2:
        print(z[0])
        print(h[0])
        
        y_hat = W[1] @ h[1] + b[1]
        print(y_hat)
        loss = half_mse(y_hat, y, W, alpha)
    elif len(W)== 1:
        y_hat = z[0]
        loss = half_mse(y_hat, y, W, alpha)
    return loss, z, h, x, y_hat

def back_prop(X, y, W: list[np.ndarray], B:list[np.ndarray], alpha, act_fn):
    
    print(f' fwd prop: {forward_prop(X, y, W, B, alpha, act_fn)}')
    loss, z, h, x, yhat = forward_prop(X, y, W, B, alpha, act_fn)
    print(f' loss from fprop: {loss}')
    print(f' h from fprop: {h}')
    print(f' pred from fprop: {yhat}')

    X_col = x.reshape(-1, 1)
    yhat = yhat.reshape(-1, 1)
    # print(h)
    dW = []
    dB = []
    # for k, v in h.items():
    #     print(k)
    dfdyhat = (y - yhat)
    dout = dfdyhat    
    n_layers = len(W)
    delta = None
    for i in range(n_layers -1, -1, -1):
        if i == n_layers - 1:
            delta = dout
        else:
            if act_fn == relu:
                delta = delta * (z[i] > 0)
            elif act_fn == sigmoid:
                delta = delta * (h[i] * (1 - h[i]))
            elif act_fn == tanh:
                delta = delta * (1 - h[i]**2)
            else:
                print('wrong act fn')
                return None
        if i > 0:
            print(delta)
            dW_i = delta @ h[i-1].T
        else:
            dW_i = delta@X_col.T

        dB_i = delta.reshape(-1)
        dW.insert(0, dW_i)
        dB.insert(0, dB_i)

        if i > 0:
            delta = W[i].T @ delta

    return dW, dB, loss

if __name__ == "__main__":
    x = np.array([1,2])
    y = np.array([0])
    W1 = np.random.rand(4, x.shape[-1])
    b1 = np.random.rand(4)
    W2 = np.random.rand(3, 4)
    b2 = np.random.rand(3)
    W3 = np.random.rand(2, 3)
    b3 = np.random.rand(2)
    W4 = np.random.rand(1, 2)
    b4 = np.random.rand(1)
    W = [W1, W2, W3, W4]
    b =  [b1, b2, b3, b4] 
    # print(forward_prop(x, y, [W1, W2, W3, W4], [b1, b2, b3, b4], 0.1, relu))
    gradients_w, gradients_b, loss = back_prop(x, y, W, b, 0.1, relu )
    for w in range(len(gradients_w)):
        print(f' layer of w: {(len(W)-w)}, gradient : {gradients_w[w]}')
    

