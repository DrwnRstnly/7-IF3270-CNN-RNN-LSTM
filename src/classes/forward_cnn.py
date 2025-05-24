import h5py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from classes.Layer import Layer

def load_weights(h5_path):
    weights = {}
    with h5py.File(h5_path, 'r') as f:
        layers_grp = f['layers']
        for layer_name, layer_grp in layers_grp.items():
            vars_grp = layer_grp.get('vars')
            if vars_grp is not None and set(vars_grp.keys()) >= {'0', '1'}:
                W = vars_grp['0'][()]
                b = vars_grp['1'][()]
                weights[layer_name] = (W, b)
    return weights

class Conv2DLayer:
    def __init__(self, W, b):
        self.W, self.b = W, b

    def forward(self, X):
        batch, h, w, c_in = X.shape
        kh, kw, _, c_out = self.W.shape
        pad_h, pad_w = kh//2, kw//2
        Xp = np.pad(X, ((0,0),(pad_h,pad_h),(pad_w,pad_w),(0,0)), mode='constant')
        out = np.zeros((batch, h, w, c_out), dtype=X.dtype)
        for bi in range(batch):
            for i in range(h):
                for j in range(w):
                    patch = Xp[bi, i:i+kh, j:j+kw, :]
                    for co in range(c_out):
                        out[bi,i,j,co] = np.sum(patch * self.W[:,:,:,co]) + self.b[co]
        return np.maximum(0, out)  # ReLU

class PoolingLayer:
    def __init__(self, mode='max'):
        self.mode = mode

    def forward(self, X):
        batch, h, w, c = X.shape
        nh, nw = h//2, w//2
        out = np.zeros((batch, nh, nw, c), dtype=X.dtype)
        for bi in range(batch):
            for i in range(nh):
                for j in range(nw):
                    region = X[bi, 2*i:2*i+2, 2*j:2*j+2, :]
                    if self.mode=='max':
                        out[bi,i,j] = np.max(region, axis=(0,1))
                    else:
                        out[bi,i,j] = np.mean(region, axis=(0,1))
        return out

class FlattenLayer:
    def forward(self, X):
        return X.reshape(X.shape[0], -1)

class CNN:
    def __init__(self, weights, config):
        self.layers = []
        for spec in config:
            typ = spec[0]
            if typ=='conv':
                name = spec[1]
                W, b = weights[name]
                self.layers.append(Conv2DLayer(W, b))
            elif typ=='pool':
                mode = spec[1]
                self.layers.append(PoolingLayer(mode))
            elif typ=='flatten':
                self.layers.append(FlattenLayer())
            elif typ=='dense':
                name, act = spec[1], spec[2]
                W, b = weights[name]
                layer = Layer(input_size=W.shape[0],
                              output_size=W.shape[1],
                              activation=act,
                              weight_init='zero')
                layer.W = W
                layer.b = b.reshape(1, -1)
                self.layers.append(layer)

    def forward(self, X):
        out = X
        for l in self.layers:
            out = l.forward(out)
        return out

    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

if __name__ == '__main__':
    print('Running unit tests for forward_cnn_from_scratch.py')

    # Conv2DLayer test: shape & ReLU
    X = np.arange(16).reshape(1,4,4,1).astype(float)
    W = np.ones((3,3,1,1))
    b = np.zeros((1,))
    conv = Conv2DLayer(W, b)
    out = conv.forward(X)
    assert out.shape == (1,4,4,1)
    assert (out >= 0).all()

    # PoolingLayer test: max & average
    Y = np.arange(16).reshape(1,4,4,1).astype(float)
    pmax = PoolingLayer(mode='max')
    out_max = pmax.forward(Y)
    assert out_max.shape == (1,2,2,1)
    assert out_max[0,0,0,0] == 5
    pavg = PoolingLayer(mode='average')
    out_avg = pavg.forward(Y)
    assert np.isclose(out_avg[0,0,0,0], 2.5)

    # FlattenLayer test
    Z = np.zeros((2,3,4,5))
    flat = FlattenLayer()
    out_flat = flat.forward(Z)
    assert out_flat.shape == (2,3*4*5)

    weights = {
        'c': [np.ones((3,3,1,1)), np.zeros((1,))],
        'd': [np.ones((2*2*1,2)), np.zeros((2,))]
    }
    config = [
        ('conv','c'),
        ('pool','max'),
        ('flatten',None),
        ('dense','d','linear'),
    ]
    model = CNN(weights, config)
    Xs = np.ones((1,4,4,1))
    y_pred = model.predict(Xs)
    assert y_pred.shape == (1,)
    assert y_pred[0] in [0,1]

    print('All unit tests passed!')