''' Layers '''
import numpy as np

def forward_col_indices(x, field_height, field_width, stride, padding):
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    N, C, H, W = x.shape
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1
    i0 = np.repeat(np.arange(field_height,dtype='int32'), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height,dtype='int32'), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width,dtype='int32'), int(out_height))
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C,dtype='int32'), field_height * field_width).reshape(-1, 1)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
      
    return cols

def backward_col_indices(cols, x_shape, field_height, field_width, padding, stride):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    N, C, H, W = x_shape
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1
    i0 = np.repeat(np.arange(field_height,dtype='int32'), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height,dtype='int32'), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width,dtype='int32'), int(out_height))
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C,dtype='int32'), field_height * field_width).reshape(-1, 1)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

class Conv():

    def __init__(self, X_dim, n_filter, h_filter, w_filter, stride, padding):

        self.d_X, self.h_X, self.w_X = X_dim
        self.n_filter, self.h_filter, self.w_filter = n_filter, h_filter, w_filter
        self.stride, self.padding = stride, padding
        self.W = np.random.randn(n_filter, self.d_X, h_filter, w_filter) / np.sqrt(n_filter / 2.)
        self.b = np.zeros((self.n_filter, 1))
        self.params = [self.W, self.b]
        self.h_out = (self.h_X - h_filter + 2 * padding) / stride + 1
        self.w_out = (self.w_X - w_filter + 2 * padding) / stride + 1
        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")
        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.out_dim = (self.n_filter, self.h_out, self.w_out)

    def forward(self, X):
        self.n_X = X.shape[0]
        self.X_col = forward_col_indices(X, self.h_filter, self.w_filter, stride=self.stride, padding=self.padding)
        W_row = self.W.reshape(self.n_filter, -1)
        out = np.dot(W_row,self.X_col) + self.b
        out = out.reshape(self.n_filter, self.h_out, self.w_out, self.n_X)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        dout_flat = dout.transpose(1, 2, 3, 0).reshape(self.n_filter, -1)
        dW = dout_flat @ self.X_col.T
        dW = dW.reshape(self.W.shape)
        db = np.sum(dout, axis=(0, 2, 3)).reshape(self.n_filter, -1)
        W_flat = self.W.reshape(self.n_filter, -1)
        dX_col = np.dot(W_flat.T, dout_flat)
        shape = (self.n_X, self.d_X, self.h_X, self.w_X)
        dX = backward_col_indices(dX_col, shape, self.h_filter,self.w_filter, self.padding, self.stride)
        return dX, [dW, db]

class Maxpool():

    def __init__(self, X_dim, size, stride):
        self.d_X, self.h_X, self.w_X = X_dim
        self.params = []
        self.size = size
        self.stride = stride
        self.h_out = (self.h_X - size) / stride + 1
        self.w_out = (self.w_X - size) / stride + 1
        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")
        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.out_dim = (self.d_X, self.h_out, self.w_out)

    def forward(self, X):
        self.n_X = X.shape[0]
        X_reshaped = X.reshape(X.shape[0] * X.shape[1], 1, X.shape[2], X.shape[3])
        self.X_col = forward_col_indices(X_reshaped, self.size, self.size, padding=0, stride=self.stride)
        self.max_indexes = np.argmax(self.X_col, axis=0)
        out = self.X_col[self.max_indexes, range(self.max_indexes.size)]
        out = out.reshape(self.h_out, self.w_out, self.n_X,self.d_X).transpose(2, 3, 0, 1)
        return out

    def backward(self, dout):
        dX_col = np.zeros_like(self.X_col)
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()
        dX_col[self.max_indexes, range(self.max_indexes.size)] = dout_flat
        shape = (self.n_X * self.d_X, 1, self.h_X, self.w_X)
        dX = backward_col_indices(dX_col, shape, self.size,self.size, padding=0, stride=self.stride)
        dX = dX.reshape(self.n_X, self.d_X, self.h_X, self.w_X)
        return dX, []

class Flatten():

    def __init__(self):
        self.params = []

    def forward(self, X):
        self.X_shape = X.shape
        self.out_shape = (self.X_shape[0], -1)
        out = X.ravel().reshape(self.out_shape)
        self.out_shape = self.out_shape[1]
        return out

    def backward(self, dout):
        out = dout.reshape(self.X_shape)
        return out, ()

class FullyConnected():

    def __init__(self, in_size, out_size):

        self.W = np.random.randn(in_size, out_size) / np.sqrt(in_size / 2.)
        self.b = np.zeros((1, out_size))
        self.params = [self.W, self.b]

    def forward(self, X):
        self.X = X
        out = self.X @ self.W + self.b
        return out

    def backward(self, dout):
        dW = self.X.T @ dout
        db = np.sum(dout, axis=0)
        dX = dout @ self.W.T
        return dX, [dW, db]

class ReLU():
    def __init__(self):
        self.params = []

    def forward(self, X):
        self.X = X
        return np.maximum(X, 0)

    def backward(self, dout):
        dX = dout.copy()
        dX[self.X <= 0] = 0
        return dX, []

class sigmoid():

    def forward(self, X):
        out = 1.0 / (1.0 + np.exp(X))
        self.out = out
        return out

    def backward(self, dout):
        dX = dout * self.out * (1 - self.out)
        return dX, []

class tanh():

    def forward(self, X):
        out = np.tanh(X)
        self.out = out
        return out

    def backward(self, dout):
        dX = dout * (1 - self.out**2)
        return dX
    
